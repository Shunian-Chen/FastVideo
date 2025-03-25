#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import torch
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

from fastvideo.models.hunyuan.diffusion.schedulers import \
    FlowMatchDiscreteScheduler
from fastvideo.models.mochi_hf.mochi_latents_utils import normalize_dit_input
from fastvideo.utils.load import load_transformer, load_vae
from fastvideo.models.hunyuan_hf.pipeline_hunyuan import HunyuanVideoPipeline
from fastvideo.models.hunyuan.diffusion.pipelines.pipeline_hunyuan_video import HunyuanVideoAudioPipeline
from fastvideo.models.mochi_hf.pipeline_mochi import MochiPipeline
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
    parser.add_argument("--model_path", type=str, help="模型路径，如果不指定则使用异常数据中的检查点路径")
    parser.add_argument("--pretrained_model_dir", type=str, help="预训练模型路径，如果不指定则使用异常数据中的检查点路径")
    parser.add_argument("--model_type", type=str, default="hunyuan_audio", help="模型类型")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--anomaly_id", type=int, default=None, help="指定异常ID（如果指定，则只处理该ID的异常）")
    parser.add_argument("--inference_steps", type=int, default=50, help="推理步数")
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
    parser.add_argument("--precision", type=str, default="bf16", help="精度")
    return parser.parse_args()

def load_text_encoder():
    # Text encoder
    device = None
    if nccl_info.sp_size > 1:
        device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    crop_start = PROMPT_TEMPLATE["dit-llm-encode"].get(
        "crop_start", 0)
    max_length = 256 + crop_start

    # prompt_template
    prompt_template = PROMPT_TEMPLATE["dit-llm-encode"]

    # prompt_template_video
    prompt_template_video = PROMPT_TEMPLATE["dit-llm-encode-video"]

    text_encoder = TextEncoder(
        text_encoder_type='llm',
        max_length=max_length,
        text_encoder_path="/data/nas/yexin/workspace/shunian/model/text_encoder",
        text_encoder_precision="fp16",
        tokenizer_type='llm',
        prompt_template=prompt_template,
        prompt_template_video=prompt_template_video,
        hidden_state_skip_layer=2,
        apply_final_norm=True,
        reproduce=False,
        logger=logger,
        device=device,
    )

    text_encoder_2 = TextEncoder(
        text_encoder_type='clipL',
        max_length=77,
        text_encoder_path="/data/nas/yexin/workspace/shunian/model/text_encoder_2",
        text_encoder_precision="fp16",
        tokenizer_type='clipL',
        reproduce=False,
        logger=logger,
        device=device,
    )
    return text_encoder, text_encoder_2

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

def get_gpu_memory_info():
    """获取更精确的GPU内存使用信息"""
    torch.cuda.synchronize()  # 确保所有CUDA操作已完成
    
    allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # 已分配的内存（MB）
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)    # 已保留的内存（MB）
    
    # 获取峰值内存使用
    max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
    max_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)
    
    return {
        "allocated": allocated,
        "reserved": reserved,
        "max_allocated": max_allocated,
        "max_reserved": max_reserved,
    }

def print_gpu_memory_status(stage_name="当前"):
    """打印GPU内存状态信息"""
    memory_info = get_gpu_memory_info()
    
    print(f"===== {stage_name}GPU内存状态 =====")
    print(f"已分配: {memory_info['allocated']:.2f} MB")
    print(f"已保留: {memory_info['reserved']:.2f} MB")
    print(f"峰值分配: {memory_info['max_allocated']:.2f} MB")
    print(f"峰值保留: {memory_info['max_reserved']:.2f} MB")
    print("===========================")
    return memory_info

def reset_peak_memory_stats():
    """重置峰值内存统计"""
    torch.cuda.reset_peak_memory_stats()

def get_model_size(model, model_name="模型"):
    """
    计算模型的参数量和预估内存占用
    返回参数量（百万）和预估GPU占用（MB）
    """
    if model is None:
        return 0, 0
        
    # 计算参数量
    num_params = sum(p.numel() for p in model.parameters())
    
    # 计算参数占用的内存（考虑数据类型）
    param_size = 0
    for param in model.parameters():
        # 根据数据类型计算每个参数的字节数
        if param.dtype == torch.float16 or param.dtype == torch.bfloat16:
            bytes_per_param = 2  # 半精度浮点数
        elif param.dtype == torch.float32:
            bytes_per_param = 4  # 单精度浮点数
        elif param.dtype == torch.float64:
            bytes_per_param = 8  # 双精度浮点数
        elif param.dtype == torch.int8 or param.dtype == torch.uint8:
            bytes_per_param = 1  # 8位整数
        else:
            bytes_per_param = 4  # 默认当作float32处理
        
        param_size += param.numel() * bytes_per_param
    
    # 模型梯度、优化器状态等额外开销（保守估计）
    # 推理时通常不需要梯度，但仍有一些开销
    overhead_ratio = 1.2  # 假设20%的额外开销
    
    # 计算模型占用的总内存（MB）
    estimated_memory = (param_size * overhead_ratio) / (1024 * 1024)
    
    # 打印结果
    params_millions = num_params / 1000000
    print(f"{model_name} 参数量: {params_millions:.2f}M, 预估占用: {estimated_memory:.2f} MB")
    
    return params_millions, estimated_memory

def setup_pipeline(args, model_path):
    """设置推理pipeline"""
    print(f"正在从 {model_path} 加载模型...")
    
    # 用于跟踪各组件大小的字典
    model_sizes = {
        "transformer": {"params": 0, "memory": 0},
        "text_encoder": {"params": 0, "memory": 0},
        "text_encoder_2": {"params": 0, "memory": 0},
        "vae": {"params": 0, "memory": 0},
        "total": {"params": 0, "memory": 0}
    }
    
    # 确定合适的dit_model_path
    dit_model_path = None
    if args.model_type in ["hunyuan_audio"]:
        # 对于hunyuan_audio模型，我们需要提供模型权重文件路径
        # 检查是否是checkpoint路径
        if os.path.isdir(model_path):
            # 首先查找safetensors文件（优先使用）
            transformer_path = os.path.join(model_path, "diffusion_pytorch_model.safetensors")
            if os.path.exists(transformer_path):
                dit_model_path = transformer_path
                print(f"找到safetensors模型文件: {dit_model_path}")
            else:
                # 查找各种可能的权重文件
                safetensors_files = glob(os.path.join(model_path, "*.safetensors"))
                pt_files = glob(os.path.join(model_path, "*.pt"))
                
                # 优先使用safetensors文件
                if safetensors_files:
                    dit_model_path = safetensors_files[0]
                    print(f"找到safetensors模型文件: {dit_model_path}")
                elif pt_files:
                    # 优先使用包含'model'的PT文件
                    model_pt_files = [f for f in pt_files if 'model' in os.path.basename(f).lower()]
                    if model_pt_files:
                        dit_model_path = model_pt_files[0]
                    else:
                        dit_model_path = pt_files[0]
                    print(f"找到PT模型文件: {dit_model_path}")
        else:
            # 如果直接提供了文件路径
            dit_model_path = model_path
            print(f"使用指定的模型文件: {dit_model_path}")
        
        if dit_model_path is None:
            raise ValueError(f"无法为hunyuan_audio找到合适的模型权重文件。请指定正确的模型路径。")
        
        # 检查文件是否真实存在
        if not os.path.exists(dit_model_path):
            raise FileNotFoundError(f"指定的模型权重文件不存在: {dit_model_path}")
            
        # print(f"最终使用权重文件: {dit_model_path}")
    
    print(f"加载模型: {args.model_type}, 模型路径: {model_path}, 权重路径: {dit_model_path}")
    
    # 重置峰值内存统计
    reset_peak_memory_stats()
    
    # 加载transformer
    print(f"加载transformer...")
    transformer = load_transformer(
        args.model_type,
        dit_model_path,  # 现在传递正确的dit_model_path
        model_path,
        torch.bfloat16
    )
    # 计算transformer大小
    model_sizes["transformer"]["params"], model_sizes["transformer"]["memory"] = get_model_size(transformer, "Transformer")
    model_sizes["total"]["params"] += model_sizes["transformer"]["params"]
    model_sizes["total"]["memory"] += model_sizes["transformer"]["memory"]
    
    print_gpu_memory_status("加载transformer后")

    # 加载text_encoder
    print(f"加载text_encoder...")
    text_encoder, text_encoder_2 = load_text_encoder()
    
    # 计算text_encoder大小
    model_sizes["text_encoder"]["params"], model_sizes["text_encoder"]["memory"] = get_model_size(text_encoder, "Text Encoder")
    model_sizes["total"]["params"] += model_sizes["text_encoder"]["params"]
    model_sizes["total"]["memory"] += model_sizes["text_encoder"]["memory"]
    
    # 计算text_encoder_2大小（如果存在）
    if text_encoder_2 is not None:
        model_sizes["text_encoder_2"]["params"], model_sizes["text_encoder_2"]["memory"] = get_model_size(text_encoder_2, "Text Encoder 2")
        model_sizes["total"]["params"] += model_sizes["text_encoder_2"]["params"]
        model_sizes["total"]["memory"] += model_sizes["text_encoder_2"]["memory"]
    
    print_gpu_memory_status("加载text_encoder后")
    
    # 加载VAE
    print(f"加载VAE...")
    vae, autocast_type, fps = load_vae(args.model_type, args.pretrained_model_dir)
    
    # 计算VAE大小
    model_sizes["vae"]["params"], model_sizes["vae"]["memory"] = get_model_size(vae, "VAE")
    model_sizes["total"]["params"] += model_sizes["vae"]["params"]
    model_sizes["total"]["memory"] += model_sizes["vae"]["memory"]
    
    print_gpu_memory_status("加载VAE后")
    
    # 打印总模型大小信息
    print(f"\n===== 模型总体大小统计 =====")
    print(f"Transformer: {model_sizes['transformer']['params']:.2f}M 参数, 预估显存: {model_sizes['transformer']['memory']:.2f} MB")
    print(f"Text Encoder: {model_sizes['text_encoder']['params']:.2f}M 参数, 预估显存: {model_sizes['text_encoder']['memory']:.2f} MB")
    if text_encoder_2 is not None:
        print(f"Text Encoder 2: {model_sizes['text_encoder_2']['params']:.2f}M 参数, 预估显存: {model_sizes['text_encoder_2']['memory']:.2f} MB")
    print(f"VAE: {model_sizes['vae']['params']:.2f}M 参数, 预估显存: {model_sizes['vae']['memory']:.2f} MB")
    print(f"总计: {model_sizes['total']['params']:.2f}M 参数, 预估显存: {model_sizes['total']['memory']:.2f} MB")
    print(f"注意: 实际GPU内存使用可能会因缓存、临时变量等原因更高")
    print("=============================\n")
    
    # 设置调度器
    scheduler = FlowMatchDiscreteScheduler(
        shift=args.flow_shift,
        reverse=args.flow_reverse,
        solver=args.flow_solver,
    )
    
    # 创建Pipeline
    pipeline = HunyuanVideoAudioPipeline(
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        transformer=transformer,
        scheduler=scheduler,
        args=args,
    )
    
    # 存储模型大小信息到pipeline对象，方便后续使用
    pipeline._model_sizes = model_sizes
    
    return pipeline

def save_video(frames, output_path, fps=24):
    """保存帧序列为视频"""
    # 确保帧是uint8格式且形状为[T,H,W,C]
    if isinstance(frames, list):
        # PIL图像列表转为视频张量 [T,H,W,C]
        video_frames = []
        for img in frames:
            if isinstance(img, Image.Image):
                video_frames.append(torch.tensor(np.array(img)))
        frames = torch.stack(video_frames)
    elif isinstance(frames, torch.Tensor):
        if frames.shape[1] == 3:  # 如果是[T,C,H,W]格式
            frames = frames.permute(0, 2, 3, 1)
    
    if frames.dtype != torch.uint8:
        if frames.max() <= 1.0:
            frames = (frames * 255).to(torch.uint8)
        else:
            frames = frames.to(torch.uint8)
    
    try:
        torchvision.io.write_video(output_path, frames, fps=fps)
        print(f"视频保存到 {output_path}")
    except Exception as e:
        print(f"保存视频时出错: {e}")
        # 尝试保存为单独的帧
        frame_dir = output_path.replace(".mp4", "_frames")
        os.makedirs(frame_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            Image.fromarray(frame.numpy()).save(os.path.join(frame_dir, f"frame_{i:04d}.png"))
        print(f"无法保存视频，已将帧保存到 {frame_dir}")

def run_inference_hunyuan(pipeline, audio_embeds, prompt, args):
    """运行推理（Hunyuan原生模型）"""
    infer_args = {
        "prompt": prompt,
        "height": args.height,
        "width": args.width, 
        "video_length": args.num_frames,
        "infer_steps": args.inference_steps,
        "guidance_scale": args.guidance_scale,
        "flow_shift": args.flow_shift,
    }
    
    if args.embedded_guidance_scale is not None:
        infer_args["embedded_guidance_scale"] = args.embedded_guidance_scale
        
    if audio_embeds is not None:
        # 对于支持音频的模型，传递音频嵌入
        if isinstance(pipeline, HunyuanVideoAudioPipeline):
            infer_args["audio_embeds"] = audio_embeds
    
    # 显存追踪列表
    memory_tracking = []
    
    # 运行推理
    start_time = time.time()
    
    # 重置内存峰值统计
    reset_peak_memory_stats()
    init_mem = get_gpu_memory_info()
    memory_tracking.append(("开始推理", init_mem))
    
    # 执行推理
    result = pipeline(**infer_args)
    
    # 记录推理后内存
    end_mem = get_gpu_memory_info()
    memory_tracking.append(("完成推理", end_mem))
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    # 打印推理性能信息
    print(f"\n===== 推理性能统计 =====")
    print(f"推理总耗时: {inference_time:.2f}秒")
    print(f"平均每帧耗时: {inference_time / args.num_frames:.2f}秒")
    print(f"帧率: {args.num_frames / inference_time:.2f} FPS")
    
    # 打印内存追踪信息
    print(f"\n===== 推理阶段内存追踪 =====")
    for stage, mem in memory_tracking:
        print(f"{stage}:")
        print(f"  已分配: {mem['allocated']:.2f} MB")
        print(f"  已保留: {mem['reserved']:.2f} MB")
    
    # 计算峰值和增长
    peak_memory = end_mem["max_allocated"]
    memory_growth = end_mem["allocated"] - init_mem["allocated"]
    print(f"\n峰值内存: {peak_memory:.2f} MB")
    print(f"内存增长: {memory_growth:.2f} MB")
    print("===========================\n")
    
    return result["samples"]

def run_inference_diffusers(pipeline, audio_embeds, prompt, args):
    """运行推理（Diffusers模型）"""
    infer_args = {
        "prompt": prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.inference_steps,
        "guidance_scale": args.guidance_scale,
        "output_type": "pt",  # 返回张量
    }
    
    if audio_embeds is not None:
        # 对于Diffusers模型，根据pipeline类型处理音频嵌入
        if isinstance(pipeline, HunyuanVideoAudioPipeline):
            infer_args["audio_embeds"] = audio_embeds
    
    # 显存追踪列表
    memory_tracking = []
    
    # 运行推理
    start_time = time.time()
    
    # 重置内存峰值统计
    reset_peak_memory_stats()
    init_mem = get_gpu_memory_info()
    memory_tracking.append(("开始推理", init_mem))
    
    # 执行推理
    result = pipeline(**infer_args)
    
    # 记录推理后内存
    end_mem = get_gpu_memory_info()
    memory_tracking.append(("完成推理", end_mem))
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    # 打印推理性能信息
    print(f"\n===== 推理性能统计 =====")
    print(f"推理总耗时: {inference_time:.2f}秒")
    print(f"平均每帧耗时: {inference_time / args.num_frames:.2f}秒")
    print(f"帧率: {args.num_frames / inference_time:.2f} FPS")
    
    # 打印内存追踪信息
    print(f"\n===== 推理阶段内存追踪 =====")
    for stage, mem in memory_tracking:
        print(f"{stage}:")
        print(f"  已分配: {mem['allocated']:.2f} MB")
        print(f"  已保留: {mem['reserved']:.2f} MB")
    
    # 计算峰值和增长
    peak_memory = end_mem["max_allocated"]
    memory_growth = end_mem["allocated"] - init_mem["allocated"]
    print(f"\n峰值内存: {peak_memory:.2f} MB")
    print(f"内存增长: {memory_growth:.2f} MB")
    print("===========================\n")
    
    if hasattr(result, "videos"):
        return result.videos
    elif hasattr(result, "frames"):
        return result.frames
    else:
        return result[0]  # 假设第一个元素是视频帧

def main():
    args = parse_args()
    
    # 初始化分布式
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        print(f"local_rank: {local_rank}, world_size: {world_size}")
        torch.cuda.set_device(local_rank)
        if dist.is_available() and not dist.is_initialized():
            dist.init_process_group("nccl", init_method="env://", world_size=world_size, rank=local_rank)
        initialize_sequence_parallel_state(args.sp_size)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    

    # 加载异常数据
    anomalies = load_anomaly_data(args.anomaly_dir, args.anomaly_id)
    print(f"加载了 {len(anomalies)} 条异常记录")
    
    if len(anomalies) == 0:
        print("没有找到异常记录，退出")
        return
    
    for i, anomaly in enumerate(anomalies):
        print(f"\n处理异常 {i+1}/{len(anomalies)} - 步骤 {anomaly['step']}")
        
        # 决定使用哪个模型检查点
        model_path = args.model_path
        if model_path is None and anomaly.get("model_checkpoint"):
            model_path = anomaly["model_checkpoint"]
        
        if model_path is None:
            print(f"警告: 没有为异常步骤 {anomaly['step']} 指定模型检查点，跳过")
            continue
        
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
        
        # 设置Pipeline
        reset_peak_memory_stats()  # 重置统计信息
        pipeline = setup_pipeline(args, model_path)
        
        # 分析模型内存需求
        if hasattr(pipeline, "_model_sizes"):
            model_sizes = pipeline._model_sizes
            print("\n===== 模型内存需求分析 =====")
            print(f"当前CPU上模型总参数量: {model_sizes['total']['params']:.2f}M")
            print(f"预估GPU内存需求: {model_sizes['total']['memory']:.2f} MB")
            
            # 分析不同精度下的内存需求
            precision_factor = {
                "bfloat16/float16": 1.0,  # 基准（当前精度）
                "float32": 2.0,           # bfloat16->float32: 内存翻倍
                "int8": 0.5               # bfloat16->int8: 内存减半
            }
            
            print("\n不同精度下的预估内存需求:")
            for prec_name, factor in precision_factor.items():
                adjusted_memory = model_sizes['total']['memory'] * factor
                print(f"  {prec_name}: {adjusted_memory:.2f} MB")
            
            # 分析推理时的额外内存需求
            print("\n推理时可能的额外内存需求:")
            print(f"  激活值和中间结果: ~{model_sizes['total']['memory'] * 0.5:.2f} MB")
            print(f"  注意力缓存: ~{model_sizes['transformer']['memory'] * 0.3:.2f} MB")
            print(f"  批次大小为1的总预估内存: ~{model_sizes['total']['memory'] * 1.8:.2f} MB")
            print("=============================\n")
        
        # 记录to(device)前的内存状态
        # print_gpu_memory_status("pipeline to device 前")
        # before_to_device = get_gpu_memory_info()
        
        # 移动模型到设备
        # print(f"移动模型到设备...")
        # print(f"显存占用: {get_gpu_memory_info()['allocated']:.2f} MB")
        # pipeline.vae.to(device)
        # print(f"移动vae到设备后显存占用: {get_gpu_memory_info()['allocated']:.2f} MB, vae数据类型: {pipeline.vae.dtype}")
        # pipeline.text_encoder.to(device)
        # print(f"移动text_encoder到设备后显存占用: {get_gpu_memory_info()['allocated']:.2f} MB, text_encoder数据类型: {pipeline.text_encoder.dtype}")
        # pipeline.text_encoder_2.to(device)
        # print(f"移动text_encoder_2到设备后显存占用: {get_gpu_memory_info()['allocated']:.2f} MB, text_encoder_2数据类型: {pipeline.text_encoder_2.dtype}")
        # pipeline.transformer.to(device)
        # print(f"移动transformer到设备后显存占用: {get_gpu_memory_info()['allocated']:.2f} MB, transformer数据类型: {pipeline.transformer.dtype}")
        # pipeline.to(device)
        # print(f"移动pipeline到设备后显存占用: {get_gpu_memory_info()['allocated']:.2f} MB, pipeline数据类型: {pipeline.dtype}")
        
        # 记录to(device)后的内存状态
        # after_to_device = get_gpu_memory_info()
        # print_gpu_memory_status("pipeline to device 后")
        
        # 计算实际增加的显存
        actual_increase = after_to_device["allocated"] - before_to_device["allocated"]
        if hasattr(pipeline, "_model_sizes"):
            estimated = pipeline._model_sizes["total"]["memory"]
            accuracy = (actual_increase / estimated) * 100 if estimated > 0 else 0
            print(f"\n===== 显存使用分析 =====")
            print(f"预估显存: {estimated:.2f} MB")
            print(f"实际增加显存: {actual_increase:.2f} MB")
            print(f"预估准确率: {accuracy:.2f}%")
            print(f"差异: {actual_increase - estimated:.2f} MB ({'+' if actual_increase > estimated else '-'}{abs((actual_increase - estimated) / estimated) * 100 if estimated > 0 else 0:.2f}%)")
            print("===========================\n")
        
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
            if audio_embeds is not None:
                audio_embeds = audio_embeds.to(device)
            
            # 记录推理前内存
            print_gpu_memory_status("推理前")
            inference_before = get_gpu_memory_info()
            
            # 根据模型类型选择推理方法
            if args.model_type in ["hunyuan", "hunyuan_audio"] and not args.model_type.endswith("_hf"):
                frames = run_inference_hunyuan(pipeline, audio_embeds, prompt, args)
            else:
                frames = run_inference_diffusers(pipeline, audio_embeds, prompt, args)
            
            # 记录推理后内存
            inference_after = get_gpu_memory_info()
            print_gpu_memory_status("推理后")
            
            # 计算推理过程中的峰值内存
            inference_peak = inference_after["max_allocated"] - before_to_device["allocated"]
            print(f"\n===== 推理阶段显存分析 =====")
            print(f"推理前显存: {inference_before['allocated']:.2f} MB")
            print(f"推理后显存: {inference_after['allocated']:.2f} MB")
            print(f"推理过程峰值显存: {inference_peak:.2f} MB")
            if hasattr(pipeline, "_model_sizes"):
                estimated_inference = pipeline._model_sizes["total"]["memory"] * 1.8  # 使用前面估计的1.8倍因子
                inference_accuracy = (inference_peak / estimated_inference) * 100 if estimated_inference > 0 else 0
                print(f"预估推理显存: {estimated_inference:.2f} MB")
                print(f"预估准确率: {inference_accuracy:.2f}%")
            print("=============================\n")
            
            # 保存视频
            save_video(frames, output_video_path, fps=args.fps)
            
        except Exception as e:
            print(f"推理失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 释放内存
        del pipeline
        torch.cuda.empty_cache()
        print_gpu_memory_status("释放内存后")
    
    # 清理分布式组
    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()

if __name__ == "__main__":
    main() 