import os
import argparse
import random
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import Lambda
from torchvision.utils import save_image
from einops import rearrange
from pathlib import Path

from fastvideo.dataset.transform import (CenterCropResizeVideo, Normalize255,
                                         TemporalRandomCrop)
from fastvideo.dataset import getdataset


def parse_args():
    parser = argparse.ArgumentParser(description='测试视频transform前后的差异')
    parser.add_argument('--video_path', type=str, required=True, help='输入视频路径')
    parser.add_argument('--output_dir', type=str, default='transform_test_output', help='输出目录')
    parser.add_argument('--num_frames', type=int, default=16, help='采样帧数')
    parser.add_argument('--max_height', type=int, default=320, help='最大高度')
    parser.add_argument('--max_width', type=int, default=240, help='最大宽度')
    parser.add_argument('--cache_dir', type=str, default='../cache_dir', help='缓存目录')
    parser.add_argument('--text_encoder_name', type=str, default='google/mt5-xxl', help='文本编码器名称')
    return parser.parse_args()


def setup_args(args):
    """设置与getdataset兼容的参数对象"""
    class Args:
        def __init__(self):
            self.dataset = "t2v"
            self.num_frames = args.num_frames
            self.max_height = args.max_height
            self.max_width = args.max_width
            self.cache_dir = args.cache_dir
            self.text_encoder_name = args.text_encoder_name
            self.text_max_length = 300
            self.use_image_num = 0
            self.train_fps = 24
            self.drop_short_ratio = 1.0
            self.use_img_from_vid = False
            self.speed_factor = 1.0
            self.cfg = 0.1
            self.image_data = ""
            self.video_data = "1"
    
    return Args()


def save_frames(frames, output_dir, prefix="frame", normalize=True):
    """保存帧到指定目录"""
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        if normalize:
            # 确保值在[0,1]范围内
            if frame.max() > 1.0:
                frame = frame / 255.0
        else:
            # 如果是[-1,1]范围，转换到[0,1]
            if frame.min() < 0:
                frame = (frame + 1.0) / 2.0
        
        save_image(
            frame,
            os.path.join(output_dir, f"{prefix}_{i:04d}.png")
        )


def save_video(frames, output_path, fps=24):
    """保存帧序列为视频"""
    # 确保帧是uint8格式且形状为[T,H,W,C]
    if frames.shape[1] == 3:  # 如果是[T,C,H,W]格式
        frames = frames.permute(0, 2, 3, 1)
    
    if frames.dtype != torch.uint8:
        if frames.max() <= 1.0:
            frames = (frames * 255).to(torch.uint8)
        else:
            frames = frames.to(torch.uint8)
    
    torchvision.io.write_video(output_path, frames, fps=fps)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建与getdataset兼容的参数对象
    dataset_args = setup_args(args)
    
    # 获取transform
    temporal_sample = TemporalRandomCrop(dataset_args.num_frames)
    norm_fun = Lambda(lambda x: 2.0 * x - 1.0)
    
    # 定义不同的transform
    resize = [
        CenterCropResizeVideo((dataset_args.max_height, dataset_args.max_width)),
    ]
    resize_topcrop = [
        CenterCropResizeVideo((dataset_args.max_height, dataset_args.max_width), top_crop=True),
    ]
    
    # 创建不同的transform组合
    transform_basic = transforms.Compose([
        *resize,
    ])
    
    transform_normalized = transforms.Compose([
        Normalize255(),
        *resize,
    ])
    
    transform_normalized_norm_fun = transforms.Compose([
        Normalize255(),
        *resize,
        norm_fun,
    ])
    
    transform_topcrop = transforms.Compose([
        Normalize255(),
        *resize_topcrop,
        norm_fun,
    ])
    
    # 读取视频
    print(f"读取视频: {args.video_path}")
    vframes, aframes, info = torchvision.io.read_video(
        filename=args.video_path, 
        pts_unit="sec",
        output_format="TCHW"
    )
    
    # 采样帧
    total_frames = len(vframes)
    print(f"视频总帧数: {total_frames}")
    
    # 使用TemporalRandomCrop采样
    start_frame_ind, end_frame_ind = temporal_sample(total_frames)
    frame_indices = np.linspace(start_frame_ind, end_frame_ind - 1, dataset_args.num_frames, dtype=int)
    print(f"采样帧索引: {frame_indices}")
    
    # 获取采样帧
    sampled_frames = vframes[frame_indices]
    print(f"采样帧形状: {sampled_frames.shape}")
    
    # 保存原始帧
    original_dir = os.path.join(args.output_dir, "original")
    save_frames(sampled_frames, original_dir)
    save_video(sampled_frames, os.path.join(args.output_dir, "original.mp4"))
    
    # 应用不同的transform并保存结果
    transforms_to_test = {
        "basic": transform_basic,
        "normalized": transform_normalized,
        "normalized_norm_fun": transform_normalized_norm_fun,
        "topcrop": transform_topcrop
    }
    
    results = {}
    
    for name, transform in transforms_to_test.items():
        print(f"应用transform: {name}")
        transformed_frames = transform(sampled_frames.clone())
        print(f"Transform后形状: {transformed_frames.shape}")
        
        # 保存结果
        output_dir = os.path.join(args.output_dir, name)
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存帧
        is_normalized = "normalized" in name or "topcrop" in name
        save_frames(transformed_frames, output_dir, normalize=not is_normalized)
        
        # 保存视频
        if is_normalized and "norm_fun" in name:
            # 如果是[-1,1]范围，转换回[0,1]再保存
            video_frames = (transformed_frames + 1.0) / 2.0 * 255
        elif is_normalized:
            # 如果是[0,1]范围，转换回[0,255]
            video_frames = transformed_frames * 255
        else:
            video_frames = transformed_frames
            
        save_video(video_frames, os.path.join(args.output_dir, f"{name}.mp4"))
        
        results[name] = transformed_frames
    
    # 创建比较图
    plt.figure(figsize=(15, 10))
    
    # 原始图像
    plt.subplot(2, 3, 1)
    plt.title("原始帧")
    plt.imshow(sampled_frames[0].permute(1, 2, 0).numpy() / 255.0)
    plt.axis('off')
    
    # 基本transform
    plt.subplot(2, 3, 2)
    plt.title("基本transform")
    plt.imshow(results["basic"][0].permute(1, 2, 0).numpy() / 255.0)
    plt.axis('off')
    
    # Normalize255
    plt.subplot(2, 3, 3)
    plt.title("Normalize255")
    plt.imshow(results["normalized"][0].permute(1, 2, 0).numpy())
    plt.axis('off')
    
    # Normalize255 + norm_fun
    plt.subplot(2, 3, 4)
    plt.title("Normalize255 + norm_fun")
    plt.imshow((results["normalized_norm_fun"][0].permute(1, 2, 0).numpy() + 1.0) / 2.0)
    plt.axis('off')
    
    # Top crop
    plt.subplot(2, 3, 5)
    plt.title("Top crop")
    plt.imshow((results["topcrop"][0].permute(1, 2, 0).numpy() + 1.0) / 2.0)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "comparison.png"))
    
    print(f"测试完成！结果保存在: {args.output_dir}")
    print(f"- 原始视频: {os.path.join(args.output_dir, 'original.mp4')}")
    for name in transforms_to_test.keys():
        print(f"- {name}视频: {os.path.join(args.output_dir, f'{name}.mp4')}")
    print(f"- 对比图: {os.path.join(args.output_dir, 'comparison.png')}")


if __name__ == "__main__":
    main() 