#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import torch
import hashlib
import imageio
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from collections import defaultdict
from PIL import Image
from pathlib import Path
import datetime
import torchvision
import time
from loguru import logger
from einops import rearrange

from fastvideo.models.hunyuan.diffusion.schedulers import \
    FlowMatchDiscreteScheduler
from fastvideo.models.mochi_hf.mochi_latents_utils import normalize_dit_input
from fastvideo.utils.load import load_transformer, load_vae
from fastvideo.models.hunyuan.inference import HunyuanAudioVideoSampler

from fastvideo.utils.communications import broadcast
from fastvideo.utils.parallel_states import (
    get_sequence_parallel_state,
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group
)
from fastvideo.models.hunyuan.constants import (NEGATIVE_PROMPT,
                                                PRECISION_TO_TYPE,
                                                PROMPT_TEMPLATE)
from fastvideo.models.hunyuan.text_encoder import TextEncoder
from fastvideo.utils.parallel_states import nccl_info

def parse_args():
    parser = argparse.ArgumentParser(description="从异常数据中推理视频")
    parser.add_argument("--anomaly_dir", type=str, required=True, help="异常数据目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--output_path", type=str, help="输出视频路径，默认为output_dir")
    parser.add_argument("--model_path", type=str, help="模型路径，如果不指定则使用异常数据中的检查点路径")
    parser.add_argument("--pretrained_model_dir", type=str, help="预训练模型路径，如果不指定则使用异常数据中的检查点路径")
    parser.add_argument("--model_type", type=str, default="hunyuan_audio", help="模型类型")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--anomaly_id", type=int, default=None, help="指定异常ID（如果指定，则只处理该ID的异常）")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="推理步数")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="引导比例")
    parser.add_argument("--width", type=int, default=1280, help="视频宽度")
    parser.add_argument("--height", type=int, default=720, help="视频高度")
    parser.add_argument("--num_frames", type=int, default=125, help="视频帧数")
    parser.add_argument("--fps", type=int, default=24, help="视频帧率")
    parser.add_argument("--embedded_guidance_scale", type=float, default=None, help="嵌入式引导比例")
    parser.add_argument("--flow_shift", type=float, default=5.0, help="Flow shift参数")
    parser.add_argument("--flow_reverse", action="store_true", help="是否反向Flow")
    parser.add_argument("--local_rank", type=int, default=0, help="本地排名")
    parser.add_argument("--sp_size", type=int, default=1, help="序列并行大小")
    parser.add_argument(
        "--flow-reverse",
        action="store_true",
        help="If reverse, learning/sampling from t=1 -> t=0.",
    )
    parser.add_argument("--flow-solver",
                    type=str,
                    default="euler",
                    help="Solver for flow matching.")
    parser.add_argument(
        "--load-key",
        type=str,
        default="module",
        help="Key to load the model states. 'module' for the main model, 'ema' for the EMA model.",
    )
    parser.add_argument(
        "--use-cpu-offload",
        action="store_true",
        help="Use CPU offload for the model load.",
    )
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--neg_prompt", type=str, default=NEGATIVE_PROMPT, help="负提示词")
    parser.add_argument("--num_videos", type=int, default=1, help="生成视频数量")
    parser.add_argument("--batch_size", type=int, default=1, help="批量大小")
    parser.add_argument("--embedded_cfg_scale", type=float, default=6.0, help="嵌入式引导比例")
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--model", type=str, default="HYVideo-T/2-cfgdistill")
    parser.add_argument("--rope-theta", type=int, default=256, help="Theta used in RoPE.")
    parser.add_argument("--vae", type=str, default="884-16c-hy")
    parser.add_argument("--vae-precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--vae-tiling", action="store_true", default=True)
    parser.add_argument("--vae-sp", action="store_true", default=False)
    parser.add_argument("--text-encoder", type=str, default="llm")
    parser.add_argument(
        "--text-encoder-precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
    )
    parser.add_argument("--text-states-dim", type=int, default=4096)
    parser.add_argument("--text-len", type=int, default=256)
    parser.add_argument("--tokenizer", type=str, default="llm")
    parser.add_argument("--prompt-template", type=str, default="dit-llm-encode")
    parser.add_argument("--prompt-template-video", type=str, default="dit-llm-encode-video")
    parser.add_argument("--hidden-state-skip-layer", type=int, default=2)
    parser.add_argument("--apply-final-norm", action="store_true")
    parser.add_argument("--text-states-dim-2", type=int, default=768)
    parser.add_argument("--tokenizer-2", type=str, default="clipL")
    parser.add_argument("--text-len-2", type=int, default=77)
    parser.add_argument("--text-encoder-2", type=str, default="clipL")
    parser.add_argument(
        "--text-encoder-precision-2",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
    )
    parser.add_argument(
        "--enable_torch_compile",
        action="store_true",
        help="Use torch.compile for speeding up STA inference without teacache",
    )
    parser.add_argument("--dit-weight", type=str, default=None, help="DIT权重路径")
    parser.add_argument(
        "--reproduce",
        action="store_true",
        help="Enable reproducibility by setting random seeds and deterministic algorithms.",
    )
    parser.add_argument(
        "--disable-autocast",
        action="store_true",
        help="Disable autocast for denoising loop and vae decoding in pipeline sampling.",
    )
    parser.add_argument(
        "--denoise-type",
        type=str,
        default="flow",
        choices=["flow"],
        help="Denoising type (flow matching etc.).",
    )
    return parser.parse_args()

def load_anomaly_data(anomaly_dir, anomaly_id=None):
    """加载异常数据记录"""
    jsonl_files = glob(os.path.join(anomaly_dir, "anomalies_rank_*.jsonl"))
    all_anomalies = []
    seen_steps = set()
    
    for file_path in jsonl_files:
        with open(file_path, "r") as f:
            for line in f:
                anomaly = json.loads(line.strip())
                step = anomaly["step"]
                
                # 如果指定了anomaly_id，只处理该ID的异常
                if anomaly_id is not None and step != anomaly_id:
                    continue
                
                # 去除重复的step
                if step not in seen_steps:
                    seen_steps.add(step)
                    all_anomalies.append(anomaly)
    
    # 按步骤排序
    all_anomalies.sort(key=lambda x: x["step"])
    return all_anomalies

def load_batch_data(anomaly, base_dir):
    """加载批次数据"""
    data_dir = os.path.join(base_dir, anomaly["data_dir"])
    batch_data = {}
    
    for key, value in anomaly["batch_info"].items():
        if key.endswith("_path"):
            data_key = key[:-5]  # 去掉'_path'后缀
            file_path = os.path.join(base_dir, value)
            if os.path.exists(file_path):
                batch_data[data_key] = torch.load(file_path, map_location="cpu")
            else:
                print(f"警告: 文件{file_path}不存在")
        elif key.endswith("_shape"):
            continue  # 跳过形状信息
        elif isinstance(value, list) and value and isinstance(value[0], str):
            # 文件路径列表
            batch_data[key] = value
    
    return batch_data


def main():
    args = parse_args()
    
    # 标准化参数命名
    args.flow_reverse = args.flow_reverse or getattr(args, "flow-reverse", False)
    args.flow_solver = getattr(args, "flow-solver", "euler")
    args.denoise_type = getattr(args, "denoise-type", "flow")
    
    # 设置默认的输出路径
    if args.output_path is None:
        args.output_path = args.output_dir
    
    # 初始化分布式
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        print(f"local_rank: {local_rank}, world_size: {world_size}")
        torch.cuda.set_device(local_rank)
        if dist.is_available() and not dist.is_initialized():
            dist.init_process_group("nccl", init_method="env://", world_size=world_size, rank=local_rank)
        initialize_sequence_parallel_state(args.sp_size)

    models_root_path = Path(args.model_path)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    # 设置Sampler
    # sampler = HunyuanAudioVideoSampler.from_pretrained(models_root_path, args)
    sampler = None
    # 加载异常数据
    anomalies = load_anomaly_data(args.anomaly_dir, args.anomaly_id)
    print(f"加载了 {len(anomalies)} 条异常记录")
    
    if len(anomalies) == 0:
        print("没有找到异常记录，退出")
        return
    
    for i, anomaly in enumerate(anomalies):
        print(f"\n处理异常 {i+1}/{len(anomalies)} - 步骤 {anomaly['step']}")
        
        
        # 加载批次数据
        batch_data = load_batch_data(anomaly, os.path.dirname(args.anomaly_dir))
        
        # 检查批次数据是否包含必要的键
        if "audio_embeds" not in batch_data and "audio_audio_embeds" in batch_data:
            batch_data["audio_embeds"] = batch_data["audio_audio_embeds"]
        
        # 获取提示词
        prompt = "一个人在说话"  # 默认提示词
        if "audio_embed_file" in batch_data and batch_data["audio_embed_file"]:
            # 如果有音频嵌入文件，可以从中提取提示词或使用文件名
            audio_file = batch_data["audio_embed_file"][0] if isinstance(batch_data["audio_embed_file"], list) else batch_data["audio_embed_file"]
            prompt = f"一个人在说话，使用音频 {os.path.basename(audio_file)}"
        

        
        # 准备输出路径
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_video_path = os.path.join(
            args.output_dir, 
            f"anomaly_step_{anomaly['step']}_{timestamp}.mp4"
        )
        
        # 运行推理
        try:
            # 提取音频嵌入
            audio_embeds = batch_data.get("audio_embeds", None)
            print(f"audio_embeds shape before sampler: {audio_embeds.shape}")
            
            outputs = sampler.predict(
                prompt=prompt,
                height=args.height,
                width=args.width,
                video_length=args.num_frames,
                seed=args.seed,
                negative_prompt=args.neg_prompt,
                infer_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                num_videos_per_prompt=args.num_videos,
                audio_embeds=audio_embeds,
                flow_shift=args.flow_shift,
                batch_size=args.batch_size,
                embedded_guidance_scale=args.embedded_cfg_scale,
            )
            
            samples = outputs["samples"]
            # 检查samples格式并转换为视频帧
            if isinstance(samples, list):
                # 如果是PIL图像列表
                if hasattr(samples[0], 'convert'):
                    # 将PIL图像转换为numpy数组
                    frames = [np.array(img) for img in samples]
                else:
                    # 假设是tensor
                    frames = [(img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) 
                             for img in samples]
            else:
                # 如果是tensor，重新排列维度
                videos = rearrange(samples, "b c t h w -> t b c h w")
                frames = []
                for x in videos:
                    if x.shape[0] > 1:  # 批量大小 > 1
                        x = torchvision.utils.make_grid(x, nrow=min(x.shape[0], 4))
                    else:
                        x = x[0]  # 取第一个样本
                    x = x.permute(1, 2, 0).cpu().numpy()
                    frames.append((x * 255).astype(np.uint8))
            
            # 保存视频
            prompt_hash = hashlib.sha1(prompt.encode()).hexdigest()[:8]
            save_name = f"anomaly_step_{anomaly['step']}_{prompt_hash}.mp4"
            save_path = os.path.join(args.output_path, save_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            print(f"正在保存视频到: {save_path}，帧数: {len(frames)}")
            imageio.mimsave(save_path, frames, fps=args.fps)
            print(f"视频已保存: {save_path}")
            
        except Exception as e:
            print(f"推理失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 释放内存
        torch.cuda.empty_cache()
        
    # 清理分布式组
    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()

if __name__ == "__main__":
    main() 