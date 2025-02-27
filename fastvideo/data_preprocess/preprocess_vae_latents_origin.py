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


# 自定义collate函数，处理Path对象
def custom_collate_fn(batch):
    """
    自定义collate函数，将Path对象转换为字符串
    """
    elem = batch[0]
    if isinstance(elem, dict):
        return {key: custom_collate_fn([d[key] for d in batch if key in d]) for key in elem}
    elif isinstance(elem, (list, tuple)):
        return type(elem)(custom_collate_fn(samples) for samples in zip(*batch))
    elif isinstance(elem, Path):
        # 将Path对象转换为字符串
        return [str(path) for path in batch]
    else:
        # 使用默认的collate函数处理其他类型
        try:
            return torch.utils.data._utils.collate.default_collate(batch)
        except TypeError:
            # 如果默认collate失败，检查是否有Path对象
            if any(isinstance(item, Path) for item in batch):
                return [str(item) if isinstance(item, Path) else item for item in batch]
            # 如果不是Path对象导致的错误，直接返回原始batch
            return batch


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
    
    # 确保每个进程使用正确的GPU
    torch.cuda.set_device(local_rank)
    
    # 初始化分布式环境
    if not dist.is_initialized():
        try:
            dist.init_process_group(backend="nccl",
                                    init_method="env://",
                                    world_size=world_size,
                                    rank=local_rank)
            logging.info(f"[Rank {local_rank}] 分布式环境初始化成功")
        except Exception as e:
            logging.error(f"[Rank {local_rank}] 分布式环境初始化失败: {str(e)}")
            # 如果分布式初始化失败，我们仍然可以继续单进程处理
            world_size = 1
    
    logging.info(f"[Rank {local_rank}] 使用GPU: {torch.cuda.current_device()}")
    
    train_dataset = getdataset(args)
    logging.info(f"[Rank {local_rank}] 数据集大小: {len(train_dataset)}")
    
    # 如果分布式初始化成功，使用DistributedSampler
    if world_size > 1 and dist.is_initialized():
        sampler = DistributedSampler(train_dataset,
                                    rank=local_rank,
                                    num_replicas=world_size,
                                    shuffle=True)
        logging.info(f"[Rank {local_rank}] 使用分布式采样器")
    else:
        # 否则使用普通的随机采样器
        sampler = None
        logging.info(f"[Rank {local_rank}] 使用普通随机采样器")
    
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=(sampler is None),  # 如果没有sampler，则启用shuffle
        collate_fn=custom_collate_fn  # 使用自定义的collate函数
    )
    logging.info(f"[Rank {local_rank}] 数据加载器创建完成，共有 {len(train_dataloader)} 批次")

    encoder_device = torch.device(f"cuda:{local_rank}")
    
    vae, autocast_type, fps = load_vae(args.model_type, args.model_path)
    vae = vae.to(encoder_device)
    vae.enable_tiling()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "latent"), exist_ok=True)

    # 检查是否有已处理的文件
    processed_files = set()
    latent_dir = os.path.join(args.output_dir, "latent")
    for file in os.listdir(latent_dir):
        if file.endswith(".pt"):
            processed_files.add(file)
    logging.info(f"[Rank {local_rank}] 已有 {len(processed_files)} 个处理过的文件")

    json_data = []
    processed_count = 0
    skipped_count = 0
    
    for i, data in tqdm(enumerate(train_dataloader), disable=local_rank != 0):
        logging.info(f"[Rank {local_rank}] 处理批次 {i}")
        if len(data["path"]) == 0:
            logging.info(f"[Rank {local_rank}] 空批次，跳过")
            continue
        
        # 打印数据结构，帮助调试
        if i == 0:
            logging.info(f"[Rank {local_rank}] 数据结构: {', '.join(data.keys())}")
            for key, value in data.items():
                if isinstance(value, (list, tuple)):
                    logging.info(f"[Rank {local_rank}] {key} 类型: {type(value)}, 长度: {len(value)}")
                    if len(value) > 0:
                        logging.info(f"[Rank {local_rank}] {key} 第一个元素类型: {type(value[0])}")
                else:
                    logging.info(f"[Rank {local_rank}] {key} 类型: {type(value)}")
        
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
                
                # 检查文件是否已存在
                if os.path.exists(latent_path):
                    logging.info(f"[Rank {local_rank}] 文件 {latent_path} 已存在，跳过")
                    skipped_count += 1
                    
                    # 即使跳过，也添加到JSON数据中
                    item = {}
                    try:
                        # 尝试加载已存在的潜在表示以获取其形状
                        existing_latent = torch.load(latent_path, map_location="cpu")
                        item["length"] = existing_latent.shape[1]
                    except Exception as e:
                        logging.error(f"[Rank {local_rank}] 加载已存在的潜在表示时出错: {str(e)}")
                        # 如果无法加载，使用当前批次的形状
                        item["length"] = latents[idx].shape[1]
                    
                    item["latent_path"] = video_name + ".pt"
                    item["caption"] = data["text"][idx]
                    
                    # 安全地添加其他字段
                    for field in ["face_mask_path", "face_emb_path", "audio_emb_path"]:
                        if field in data and idx < len(data[field]):
                            field_value = data[field][idx]
                            # 确保字段值是字符串而不是Path对象
                            if isinstance(field_value, Path):
                                field_value = str(field_value)
                            item[field] = field_value
                    
                    json_data.append(item)
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
                    processed_count += 1
                except Exception as e:
                    logging.error(f"[Rank {local_rank}] 保存潜在表示时出错: {str(e)}")
                    continue
                
                item = {}
                item["length"] = latents[idx].shape[1]
                item["latent_path"] = video_name + ".pt"
                item["caption"] = data["text"][idx]
                
                # 安全地添加其他字段
                for field in ["face_mask_path", "face_emb_path", "audio_emb_path"]:
                    if field in data and idx < len(data[field]):
                        field_value = data[field][idx]
                        # 确保字段值是字符串而不是Path对象
                        if isinstance(field_value, Path):
                            field_value = str(field_value)
                        item[field] = field_value
                
                json_data.append(item)
                print(f"{video_name} 处理完成\n")
    
    logging.info(f"[Rank {local_rank}] 处理完成: 新处理 {processed_count} 个文件，跳过 {skipped_count} 个文件")
    
    # 每个进程保存自己的JSON数据
    rank_json_path = os.path.join(args.output_dir, f"videos2caption_rank{local_rank}.json")
    logging.info(f"[Rank {local_rank}] 保存JSON数据到 {rank_json_path}")
    try:
        with open(rank_json_path, "w") as f:
            json.dump(json_data, f, indent=4)
        logging.info(f"[Rank {local_rank}] JSON数据保存成功，共 {len(json_data)} 条记录")
    except Exception as e:
        logging.error(f"[Rank {local_rank}] 保存JSON数据时出错: {str(e)}")
    
    # 尝试使用barrier同步，但不要让它阻止程序继续执行
    sync_success = True
    if world_size > 1 and dist.is_initialized():
        try:
            dist.barrier()
            logging.info(f"[Rank {local_rank}] 所有进程同步完成")
        except Exception as e:
            logging.error(f"[Rank {local_rank}] 同步出错: {str(e)}")
            sync_success = False
    
    # 合并JSON文件
    # 如果是rank 0或者同步失败，尝试合并所有JSON文件
    if local_rank == 0 or (not sync_success and world_size == 1):
        logging.info(f"[Rank {local_rank}] 开始合并所有进程的JSON数据")
        all_json_data = []
        
        # 读取并合并所有进程的JSON文件
        for rank in range(world_size):
            rank_file = os.path.join(args.output_dir, f"videos2caption_rank{rank}.json")
            if os.path.exists(rank_file):
                try:
                    with open(rank_file, "r") as f:
                        rank_data = json.load(f)
                        if rank_data:  # 确保数据不为空
                            all_json_data.extend(rank_data)
                            logging.info(f"[Rank {local_rank}] 成功读取并合并进程 {rank} 的数据，共 {len(rank_data)} 条记录")
                        else:
                            logging.warning(f"[Rank {local_rank}] 进程 {rank} 的数据为空")
                except Exception as e:
                    logging.error(f"[Rank {local_rank}] 读取进程 {rank} 的数据时出错: {str(e)}")
        
        # 保存合并后的JSON文件
        if all_json_data:
            try:
                with open(os.path.join(args.output_dir, "videos2caption_temp.json"), "w") as f:
                    json.dump(all_json_data, f, indent=4)
                logging.info(f"[Rank {local_rank}] 合并的JSON数据保存成功，共 {len(all_json_data)} 条记录")
            except Exception as e:
                logging.error(f"[Rank {local_rank}] 保存合并的JSON数据时出错: {str(e)}")
        else:
            logging.error(f"[Rank {local_rank}] 没有有效的JSON数据可以合并")


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
