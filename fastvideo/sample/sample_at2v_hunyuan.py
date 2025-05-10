#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hunyuan Audio-to-Video 推理脚本

根据音频嵌入和文本说明生成视频。支持两种输入方式：

1. 直接指定音频嵌入文件和说明文本:
   python sample_at2v_hunyuan.py --audio_emb_path /path/to/audio.pt --caption "一个人在说英语" \
   --output_dir /path/to/output --model_path /path/to/model

2. 使用video2caption格式的JSON文件，其中包含多个数据项:
   python sample_at2v_hunyuan.py --use_v2c_format --input_path /path/to/video2caption.json \
   --output_dir /path/to/output --model_path /path/to/model

video2caption JSON 格式示例:
[
    {
        "caption": "一个男人在讲英语",
        "audio_emb_path": "/path/to/audio1.pt"
    },
    {
        "caption": "一个女人在唱歌",
        "audio_emb_path": "/path/to/audio2.pt"
    }
]

音频嵌入文件(audio_emb_path)必须是可以通过torch.load加载的张量。
"""

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
import glob
from collections import defaultdict
from PIL import Image
from pathlib import Path
import datetime
import torchvision
import time
from loguru import logger
from einops import rearrange
import subprocess

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
    parser = argparse.ArgumentParser(description="根据音频嵌入和文本描述生成视频")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--output_path", type=str, help="输出视频路径，默认为output_dir")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--pretrained_model_dir", type=str, help="预训练模型路径")
    parser.add_argument("--model_type", type=str, default="hunyuan_audio", help="模型类型")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
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
    parser.add_argument("--audio_emb_path", type=str, help="音频嵌入文件路径")
    parser.add_argument("--caption", type=str, help="视频说明文本，用于替代默认prompt")
    parser.add_argument("--use_v2c_format", action="store_true", help="是否使用video2caption数据格式")
    parser.add_argument("--input_path", type=str, help="video2caption格式的JSON文件路径")
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
    parser.add_argument("--data_dir", type=str, default=None, help="数据目录")
    parser.add_argument("--max_samples", type=int, default=None, help="要处理的最大样本数量")
    return parser.parse_args()

def process_audio_emb(audio_emb):
    # 检查是否已经有batch维度
    if len(audio_emb.shape) >= 4:  # 如果已经有至少4个维度，假设第一个是batch_size
        batch_size = audio_emb.shape[0]
        frame_num = audio_emb.shape[1]
        
        # 为每个batch单独处理
        processed_batches = []
        for b in range(batch_size):
            concatenated_tensors = []
            
            for i in range(frame_num):
                vectors_to_concat = [
                    audio_emb[b, max(min(i + j, frame_num-1), 0)] 
                    for j in range(-2, 3)]
                concatenated_tensors.append(torch.stack(vectors_to_concat, dim=0))
            
            processed_batches.append(torch.stack(concatenated_tensors, dim=0))
        
        # 在batch维度上重新组合处理过的数据
        audio_emb = torch.stack(processed_batches, dim=0)
    else:
        # 原来的处理逻辑，用于单个样本的情况
        concatenated_tensors = []
        
        for i in range(audio_emb.shape[0]):
            vectors_to_concat = [
                audio_emb[max(min(i + j, audio_emb.shape[0]-1), 0)]for j in range(-2, 3)]
            concatenated_tensors.append(torch.stack(vectors_to_concat, dim=0))
        
        audio_emb = torch.stack(concatenated_tensors, dim=0)
        # 这里处理的是单个样本，结果需要添加batch维度
        audio_emb = audio_emb.unsqueeze(0)  # 添加batch维度为1

    return audio_emb

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
    local_rank = 0
    world_size = 1
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        print(f"local_rank: {local_rank}, world_size: {world_size}")
        torch.cuda.set_device(local_rank)
        if world_size > 1 and dist.is_available() and not dist.is_initialized():
            dist.init_process_group("nccl", init_method="env://", world_size=world_size, rank=local_rank)
        if args.sp_size > 1:
             initialize_sequence_parallel_state(args.sp_size)

    models_root_path = Path(args.model_path)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    # 设置Sampler
    sampler = HunyuanAudioVideoSampler.from_pretrained(models_root_path, args)
    # sampler = None
    
    # 准备数据项列表
    data_items = []
    
    # 场景1: 使用video2caption格式的JSON文件
    if args.use_v2c_format and args.input_path:
        print(f"使用video2caption格式从{args.input_path}加载数据")
        if not os.path.exists(args.input_path):
            raise FileNotFoundError(f"video2caption JSON文件不存在: {args.input_path}")
        
        with open(args.input_path, 'r', encoding='utf-8') as f:
            data_items = json.load(f)
            
        print(f"从JSON加载了 {len(data_items)} 条数据")
    
    # 场景2: 直接使用命令行参数指定音频嵌入和说明文本
    elif args.audio_emb_path:
        if not os.path.exists(args.audio_emb_path):
            raise FileNotFoundError(f"音频嵌入文件不存在: {args.audio_emb_path}")
        
        # 创建单个数据项
        data_item = {
            "caption": args.caption or "一个人在说话",
            "audio_emb_path": args.audio_emb_path
        }
        data_items = [data_item]
        print(f"使用命令行参数创建了1条数据")
    
    # 如果没有有效数据，退出
    if not data_items:
        print("没有找到有效的数据，请使用 --use_v2c_format --input_path 或者 --audio_emb_path 参数")
        return

    # 应用 max_samples 限制 (在分片之前对总数据进行截断)
    total_items_before_limit = len(data_items)
    if args.max_samples is not None and args.max_samples > 0:
        if args.max_samples < len(data_items):
            data_items = data_items[:args.max_samples]
            print(f"已将处理样本数限制为: {args.max_samples} (原总数: {total_items_before_limit})")
        else:
            print(f"请求的最大样本数 ({args.max_samples}) 大于或等于可用样本数 ({len(data_items)})，将处理所有可用样本。")
    total_items_to_process = len(data_items) # 更新要处理的总数

    # 如果在分布式环境中运行，对数据进行分片处理
    if world_size > 1:
        # 计算每个进程应处理的数据项
        items_per_rank = (total_items_to_process + world_size - 1) // world_size  # 向上取整
        start_idx = local_rank * items_per_rank
        end_idx = min(start_idx + items_per_rank, total_items_to_process)

        # 获取当前rank应处理的数据子集
        rank_data_items = data_items[start_idx:end_idx]
        print(f"Rank {local_rank}: 将处理 {len(rank_data_items)} 个数据项 (总共 {total_items_to_process}), 索引范围: {start_idx}-{end_idx-1 if end_idx > start_idx else start_idx}")
    else:
        rank_data_items = data_items
        start_idx = 0

    # 处理每个数据项（仅处理当前进程分配的数据）
    processed_count_rank = 0 # 当前 rank 处理的计数器
    skipped_count_rank = 0 # 当前 rank 跳过的计数器
    for idx, item in enumerate(rank_data_items):
        # 计算全局索引，用于日志显示和可能的全局限制检查 (虽然限制已在分片前应用)
        global_idx = start_idx + idx if world_size > 1 else idx

        # 再次检查全局索引是否超出限制 (理论上不会，因为已在分片前截断)
        if args.max_samples is not None and args.max_samples > 0 and global_idx >= args.max_samples:
             print(f"Rank {local_rank}: 已达到全局最大样本数 {args.max_samples}，停止处理。")
             break # 如果因为某种原因之前的截断没生效，这里再加一层保险

        print(f"\nRank {local_rank}: 检查数据项 {global_idx+1}/{total_items_to_process} (本rank: {idx+1}/{len(rank_data_items)})")

        # 获取说明文本
        caption = item.get("caption", args.caption)
        if not caption:
            caption = "一个人在说话"
        # print(f"使用说明文本: {caption}") # 移动到后面，避免跳过时也打印

        # 获取音频嵌入路径
        audio_emb_path = item.get("audio_emb_path")
        if not audio_emb_path:
            print("警告: 未找到音频嵌入路径，跳过")
            skipped_count_rank += 1
            continue

        # 确保路径是绝对路径
        if not os.path.isabs(audio_emb_path):
            # 假设 audio_emb_path 是相对于 data_dir/audio_emb 的路径
            # 例如: "some_audio.pt"
            relative_audio_emb_path = audio_emb_path
            if args.data_dir:
                 audio_emb_path = os.path.join(args.data_dir, "audio_emb", relative_audio_emb_path)
            else:
                 # 如果没有 data_dir，尝试基于 input_path 的目录
                 if args.input_path:
                     base_input_dir = os.path.dirname(args.input_path)
                     audio_emb_path = os.path.join(base_input_dir, "audio_emb", relative_audio_emb_path)
                 else:
                     # 无法确定绝对路径，可能需要报错或采取默认行为
                     print(f"警告: 无法确定音频嵌入文件的绝对路径: {relative_audio_emb_path}，且未提供 --data_dir。尝试在当前目录查找。")
                     audio_emb_path = os.path.join("audio_emb", relative_audio_emb_path) # 最后尝试

        # 提取 audio_name 用于文件名检查
        audio_name = os.path.splitext(os.path.basename(item.get("audio_emb_path")))[0] # 使用原始相对路径中的文件名部分
        caption_hash = hashlib.sha1(caption.encode()).hexdigest()[:8]

        # --- 断点续传检查 ---
        # 构造用于搜索的文件名模式 (使用 audio_name, caption_hash, global_idx, 并用 * 匹配时间戳)
        search_pattern = f"{audio_name}_{caption_hash}_*_id{global_idx}.mp4"
        search_path = os.path.join(args.output_path, search_pattern)

        # 检查是否存在匹配的文件
        existing_files = glob.glob(search_path)
        if existing_files:
            print(f"Rank {local_rank}: 找到已存在的视频 {existing_files[0]} (匹配模式: {search_pattern})，跳过生成。")
            skipped_count_rank += 1
            continue # 跳到下一个 item
        # --- 检查结束 ---

        # 如果没有跳过，继续处理
        print(f"Rank {local_rank}: 处理数据项 {global_idx+1}/{total_items_to_process} (本rank: {idx+1}/{len(rank_data_items)})")
        print(f"使用说明文本: {caption}")

        # 加载音频嵌入数据
        try:
            # 检查文件是否存在
            if not os.path.exists(audio_emb_path):
                 print(f"错误: 音频嵌入文件不存在: {audio_emb_path}，跳过。")
                 skipped_count_rank += 1
                 continue

            audio_embeds = torch.load(audio_emb_path, map_location="cpu")
            if not isinstance(audio_embeds, torch.Tensor):
                audio_embeds = torch.as_tensor(audio_embeds)
            audio_embeds = process_audio_emb(audio_embeds)
            print(f"已加载音频嵌入: {audio_emb_path}")
            if hasattr(audio_embeds, "shape"):
                print(f"音频嵌入形状: {audio_embeds.shape}")
        except Exception as e:
            print(f"加载音频嵌入失败: {e}")
            skipped_count_rank += 1 # 加载失败也算跳过
            continue

        # 准备输出路径 (现在在这里生成时间戳)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # audio_name 和 caption_hash 已在前面计算
        base_filename = f"{audio_name}_{caption_hash}_{timestamp}_id{global_idx}"
        save_name = f"{base_filename}.mp4"
        temp_save_name = f"temp_{base_filename}.mp4" # 临时文件名也包含完整信息

        save_path = os.path.join(args.output_path, save_name)
        temp_video_path = os.path.join(args.output_path, temp_save_name)

        # 查找对应的音频文件
        audio_file_path = None
        if args.data_dir:
            # 使用原始相对路径中的文件名部分来查找 .wav 文件
            audio_wav_path = os.path.join(args.data_dir, "audios", f"{audio_name}.wav")
            if os.path.exists(audio_wav_path):
                audio_file_path = audio_wav_path
                print(f"找到对应的音频文件: {audio_file_path}")
            else:
                print(f"未找到对应的音频文件: {audio_wav_path}")

        # 运行推理
        try:
            outputs = sampler.predict(
                prompt=caption,
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
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 先保存无音频的视频到临时文件
            print(f"Rank {local_rank}: 正在保存临时视频到: {temp_video_path}，帧数: {len(frames)}")
            try:
                imageio.mimsave(temp_video_path, frames, fps=args.fps)
            except Exception as io_err:
                print(f"Rank {local_rank}: 保存临时视频失败: {io_err}")
                # 即使保存失败，也增加计数器，因为尝试处理了
                processed_count_rank += 1
                continue # 跳过当前项的处理

            # 如果找到了音频文件，则合并音频和视频
            if audio_file_path and os.path.exists(audio_file_path):
                # 在调用 ffmpeg 前检查临时文件是否存在
                if os.path.exists(temp_video_path):
                    try:
                        print(f"Rank {local_rank}: 正在合并音频与视频: {audio_file_path} 到 {save_path}")
                        cmd = [
                            "ffmpeg", "-y", "-v", "error",
                            "-i", temp_video_path,  # 视频输入
                            "-i", audio_file_path,  # 音频输入
                            "-c:v", "copy",         # 直接复制视频流
                            "-c:a", "aac",          # 使用AAC编码音频
                            "-shortest",            # 以最短的流长度为准
                            save_path               # 输出文件
                        ]
                        # 捕获 ffmpeg 的输出以便调试
                        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300) # 增加超时

                        # 合并成功后删除临时文件
                        if os.path.exists(temp_video_path):
                            os.remove(temp_video_path)
                        
                        print(f"Rank {local_rank}: 视频与音频已合并保存: {save_path}")
                    except subprocess.CalledProcessError as e:
                        print(f"Rank {local_rank}: 合并音频与视频失败 (ffmpeg 返回错误): {e}")
                        print(f"FFmpeg stdout: {e.stdout}")
                        print(f"FFmpeg stderr: {e.stderr}")
                        # 如果合并失败，尝试保留无音频的视频
                        if os.path.exists(temp_video_path):
                            try:
                                os.rename(temp_video_path, save_path)
                                print(f"Rank {local_rank}: 保存了无音频的视频: {save_path}")
                            except OSError as rename_err:
                                print(f"Rank {local_rank}: 重命名临时文件失败: {rename_err}. 临时文件保留在: {temp_video_path}")
                    except subprocess.TimeoutExpired:
                        print(f"Rank {local_rank}: 合并音频与视频超时。")
                        # 超时也尝试保留无音频视频
                        if os.path.exists(temp_video_path):
                             try:
                                os.rename(temp_video_path, save_path)
                                print(f"Rank {local_rank}: 保存了无音频的视频 (因合并超时): {save_path}")
                             except OSError as rename_err:
                                print(f"Rank {local_rank}: 重命名临时文件失败: {rename_err}. 临时文件保留在: {temp_video_path}")
                    except Exception as e:
                        print(f"Rank {local_rank}: 合并过程中发生未知错误: {e}")
                        # 未知错误也尝试保留无音频视频
                        if os.path.exists(temp_video_path):
                             try:
                                os.rename(temp_video_path, save_path)
                                print(f"Rank {local_rank}: 保存了无音频的视频 (因未知错误): {save_path}")
                             except OSError as rename_err:
                                print(f"Rank {local_rank}: 重命名临时文件失败: {rename_err}. 临时文件保留在: {temp_video_path}")
                else:
                    # 如果临时视频文件不存在，则无法合并
                    print(f"Rank {local_rank}: 错误：临时视频文件不存在，无法合并音频: {temp_video_path}")
                    # 此时无法保存任何视频文件，因为源文件缺失

            else:
                # 如果没有找到音频文件或音频文件不存在，检查临时文件并重命名
                if os.path.exists(temp_video_path):
                    os.rename(temp_video_path, save_path)
                    print(f"Rank {local_rank}: 视频已保存(无音频): {save_path}")
                else:
                    # 如果临时文件此时不存在，说明 imageio 保存失败或被意外删除
                    print(f"Rank {local_rank}: 错误：临时视频文件不存在，无法保存无音频视频: {temp_video_path}")
            
            # 成功处理完一个样本，增加计数器
            processed_count_rank += 1

        except Exception as e:
            print(f"Rank {local_rank}: 推理或后续处理失败: {e}")
            import traceback
            traceback.print_exc()
            # 即使失败，也增加计数器，因为尝试处理了
            # 注意：这里不应该增加 skipped_count_rank，因为我们尝试处理了但失败了
            # processed_count_rank += 1 # 移动到 try 块末尾，仅在成功时增加

        # 释放内存
        torch.cuda.empty_cache()

    print(f"Rank {local_rank}: 处理完成，成功处理了 {processed_count_rank} 个样本，跳过了 {skipped_count_rank} 个样本。")

    # 清理分布式组
    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()

if __name__ == "__main__":
    main() 