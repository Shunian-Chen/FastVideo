import argparse
import json
import os
import datetime

import torch
import torch.distributed as dist
from accelerate.logging import get_logger
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from fastvideo.dataset import getdataset
from fastvideo.utils.load import load_vae

from fastvideo.dataset.video_processor import VideoProcessor
from fastvideo.dataset.sgm.utils.audio_processor import AudioProcessor
from fastvideo.dataset.sgm.utils.image_processor import ImageProcessorForDataProcessing
from pathlib import Path
import time
import logging

logger = get_logger(__name__)


def main(args):
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size: ", world_size, "local rank: ", local_rank)
    args.gpu_rank = local_rank
    print("*"*100)
    print("args.gpu_rank", args.gpu_rank)
    print("*"*100)

    face_analysis_model_path = "./data/face_audio/face_analysis"
    landmark_model_path = "./data/face_audio/face_analysis/models/face_landmarker_v2_with_blendshapes.task"
    audio_separator_model_file = "./data/face_audio/audio_separator/Kim_Vocal_2.onnx"
    wav2vec_model_path = './data/face_audio/wav2vec/wav2vec2-base-960h'
    output_dir = Path(f"./data/Image-Vid-Finetune-HunYuan")
    
    audio_processor = AudioProcessor(
        16000,
        wav2vec_model_path,
        False,
        os.path.dirname(audio_separator_model_file),
        os.path.basename(audio_separator_model_file),
        os.path.join(output_dir, "vocals"),
    )

    image_processor = ImageProcessorForDataProcessing(
        face_analysis_model_path, landmark_model_path)
    args.video_processor = VideoProcessor(output_dir,
                                          image_processor=image_processor,
                                          audio_processor=audio_processor)
    
    torch.cuda.set_device(local_rank)
    
    if not dist.is_initialized():
        try:
            dist.init_process_group(backend="nccl",
                                    init_method="env://",
                                    world_size=world_size,
                                    rank=local_rank)
            logging.info(f"[Rank {local_rank}] 分布式环境初始化成功")
        except Exception as e:
            logging.error(f"[Rank {local_rank}] 分布式环境初始化失败: {str(e)}")
            return
    
    logging.info(f"[Rank {local_rank}] 使用GPU: {torch.cuda.current_device()}")
    
    train_dataset = getdataset(args)
    logging.info(f"[Rank {local_rank}] 数据集大小: {len(train_dataset)}")
    
    sampler = DistributedSampler(train_dataset,
                                 rank=local_rank,
                                 num_replicas=world_size,
                                 shuffle=True)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    logging.info(f"[Rank {local_rank}] 数据加载器创建完成，共有 {len(train_dataloader)} 批次")

    encoder_device = torch.device(f"cuda:{local_rank}")
    
    vae, autocast_type, fps = load_vae(args.model_type, args.model_path)
    vae = vae.to(encoder_device)
    vae.enable_tiling()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "latent"), exist_ok=True)

    json_data = []
    for i, data in tqdm(enumerate(train_dataloader), disable=local_rank != 0):
        logging.info(f"[Rank {local_rank}] 处理批次 {i}")
        if len(data["path"]) == 0:
            logging.info(f"[Rank {local_rank}] 空批次，跳过")
            continue
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=autocast_type):
                start_time = time.time()
                latents = vae.encode(data["pixel_values"].to(
                    encoder_device))["latent_dist"].sample()
                elapsed_time = time.time() - start_time
                logging.info(f"[time] {local_rank}:VAE编码和潜在采样完成，耗时 {elapsed_time:.2f} 秒")
            
            logging.info(f"[Rank {local_rank}] 处理批次中的 {len(data['path'])} 个视频")
            for idx, video_path in enumerate(data["path"]):
                logging.info(f"[Rank {local_rank}] 处理视频 {idx}: {video_path}")
                video_name = os.path.basename(video_path).split(".")[0]
                latent_path = os.path.join(args.output_dir, "latent",
                                           video_name + ".pt")
                if os.path.exists(latent_path):
                    logging.info(f"[Rank {local_rank}] 文件 {latent_path} 已存在，跳过")
                    continue
                
                video_name = os.path.basename(video_path).split(".")[0]
                timestamp = data.get("timestamp", "")
                frames = int(data.get("frames", 0))
                if isinstance(timestamp, list) and frames > 0:
                    timestamp = "_" + str(frames) + "f_" + timestamp[0]
                elif isinstance(timestamp, str) and timestamp != "" and frames > 0:
                    timestamp = "_" + str(frames) + "f_" + timestamp
                
                latent_path = os.path.join(args.output_dir, "latent",
                                           video_name + ".pt")
                
                logging.info(f"[Rank {local_rank}] 保存潜在表示到 {latent_path}")
                try:
                    torch.save(latents[idx].to(torch.bfloat16), latent_path)
                    logging.info(f"[Rank {local_rank}] 成功保存潜在表示到 {latent_path}")
                except Exception as e:
                    logging.error(f"[Rank {local_rank}] 保存潜在表示时出错: {str(e)}")
                
                item = {}
                item["length"] = latents[idx].shape[1]
                item["latent_path"] = video_name + ".pt"
                item["caption"] = data["text"][idx]
                json_data.append(item)
                print(f"{video_name} 处理完成\n")
    
    # 每个进程保存自己的JSON数据
    rank_json_path = os.path.join(args.output_dir, f"videos2caption_rank{local_rank}.json")
    logging.info(f"[Rank {local_rank}] 保存JSON数据到 {rank_json_path}")
    try:
        with open(rank_json_path, "w") as f:
            json.dump(json_data, f, indent=4)
        logging.info(f"[Rank {local_rank}] JSON数据保存成功")
    except Exception as e:
        logging.error(f"[Rank {local_rank}] 保存JSON数据时出错: {str(e)}")
    
    # 使用简单的barrier确保所有进程都完成了保存
    try:
        dist.barrier()
        logging.info(f"[Rank {local_rank}] 所有进程完成处理")
    except Exception as e:
        logging.error(f"[Rank {local_rank}] 同步出错: {str(e)}")
    
    # 只在rank 0上合并所有JSON文件
    if local_rank == 0:
        logging.info(f"[Rank {local_rank}] 开始合并所有进程的JSON数据")
        all_json_data = []
        
        # 读取并合并所有进程的JSON文件
        for rank in range(world_size):
            rank_file = os.path.join(args.output_dir, f"videos2caption_rank{rank}.json")
            if os.path.exists(rank_file):
                try:
                    with open(rank_file, "r") as f:
                        rank_data = json.load(f)
                        all_json_data.extend(rank_data)
                    logging.info(f"[Rank {local_rank}] 成功读取并合并进程 {rank} 的数据")
                except Exception as e:
                    logging.error(f"[Rank {local_rank}] 读取进程 {rank} 的数据时出错: {str(e)}")
        
        # 保存合并后的JSON文件
        try:
            with open(os.path.join(args.output_dir, "videos2caption_temp.json"), "w") as f:
                json.dump(all_json_data, f, indent=4)
            logging.info(f"[Rank {local_rank}] 合并的JSON数据保存成功")
        except Exception as e:
            logging.error(f"[Rank {local_rank}] 保存合并的JSON数据时出错: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument("--model_path", type=str, default="data/mochi")
    parser.add_argument("--model_type", type=str, default="mochi")
    parser.add_argument("--data_merge_path", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=163)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help=
        "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_latent_t",
                        type=int,
                        default=28,
                        help="Number of latent timesteps.")
    parser.add_argument("--max_height", type=int, default=480)
    parser.add_argument("--max_width", type=int, default=848)
    parser.add_argument("--video_length_tolerance_range",
                        type=int,
                        default=2.0)
    parser.add_argument("--group_frame", action="store_true")  # TODO
    parser.add_argument("--group_resolution", action="store_true")  # TODO
    parser.add_argument("--dataset", default="t2v")
    parser.add_argument("--train_fps", type=int, default=30)
    parser.add_argument("--use_image_num", type=int, default=0)
    parser.add_argument("--text_max_length", type=int, default=256)
    parser.add_argument("--speed_factor", type=float, default=1.0)
    parser.add_argument("--drop_short_ratio", type=float, default=1.0)
    # text encoder & vae & diffusion model
    parser.add_argument("--text_encoder_name",
                        type=str,
                        default="google/t5-v1_1-xxl")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=
        "The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=
        ("[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
         " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."),
    )

    args = parser.parse_args()
    main(args)
