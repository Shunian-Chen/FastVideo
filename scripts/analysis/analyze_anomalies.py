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

from fastvideo.models.mochi_hf.mochi_latents_utils import normalize_dit_input
from fastvideo.utils.load import load_transformer, load_vae
from fastvideo.utils.validation import log_validation

def parse_args():
    parser = argparse.ArgumentParser(description="分析异常数据")
    parser.add_argument("--anomaly_dir", type=str, required=True, help="异常数据目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--model_type", type=str, default="hunyuan_audio", help="模型类型")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--visualize", action="store_true", help="是否可视化")
    parser.add_argument("--analyze_specific_step", type=int, default=None, help="分析特定步骤的异常")
    return parser.parse_args()

def load_anomaly_data(anomaly_dir, specific_step=None):
    """加载所有异常数据记录"""
    jsonl_files = glob(os.path.join(anomaly_dir, "anomalies_rank_*.jsonl"))
    all_anomalies = []
    
    for file_path in jsonl_files:
        with open(file_path, "r") as f:
            for line in f:
                anomaly = json.loads(line.strip())
                if specific_step is None or anomaly["step"] == specific_step:
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
    
    return batch_data

def analyze_gradients(model, batch_data, model_type, device):
    """分析梯度"""
    # 设置模型为训练模式
    model.train()
    
    # 将数据移到设备上
    for key in batch_data:
        if isinstance(batch_data[key], torch.Tensor):
            batch_data[key] = batch_data[key].to(device)
    
    # 提取输入数据
    latents = normalize_dit_input(model_type, batch_data["latents"])
    encoder_hidden_states = batch_data["encoder_hidden_states"]
    timesteps = torch.ones(latents.shape[0], dtype=torch.long, device=device) * 500  # 中间时间步
    
    # 准备输入参数
    input_kwargs = {
        "hidden_states": latents,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": timesteps,
        "encoder_attention_mask": batch_data.get("encoder_attention_mask", None),
        "return_dict": False,
    }
    
    if "audio_embeds" in batch_data:
        input_kwargs["audio_emb"] = batch_data["audio_embeds"]
    if "face_embeds" in batch_data:
        input_kwargs["face_emb"] = batch_data["face_embeds"]
    
    if 'hunyuan' in model_type:
        input_kwargs["guidance"] = torch.tensor([1000.0], device=device, dtype=torch.bfloat16)
    
    # 确保所有梯度为零
    model.zero_grad()
    
    # 前向传播
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model_pred = model(**input_kwargs)[0]
    
    # 计算损失（简化版，实际应该使用与训练相同的损失函数）
    noise = torch.randn_like(latents)
    loss = torch.mean((model_pred - noise)**2)
    
    # 反向传播
    loss.backward()
    
    # 收集梯度信息
    gradient_info = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            grad_max = param.grad.abs().max().item()
            
            gradient_info[name] = {
                "norm": grad_norm,
                "mean": grad_mean,
                "std": grad_std,
                "max": grad_max
            }
    
    # 按梯度范数排序（降序）
    sorted_grads = sorted(gradient_info.items(), key=lambda x: x[1]["norm"], reverse=True)
    
    return {
        "loss": loss.item(),
        "all_gradients": gradient_info,
        "top_gradients": dict(sorted_grads[:20])  # 取前20个最大梯度
    }

def analyze_activations(model, batch_data, model_type, device):
    """分析激活值（需要修改模型以收集中间激活）"""
    # 这部分需要根据具体模型实现，此处仅为框架
    # 可能需要修改模型实现，添加钩子以收集中间激活值
    
    # 示例：
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # 注册钩子
    hooks = []
    for name, module in model.named_modules():
        if "attn" in name or "mlp" in name:  # 关注注意力和MLP模块
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # 前向传播（与梯度分析类似）
    # 提取输入数据
    latents = normalize_dit_input(model_type, batch_data["latents"])
    encoder_hidden_states = batch_data["encoder_hidden_states"]
    timesteps = torch.ones(latents.shape[0], dtype=torch.long, device=device) * 500  # 中间时间步
    
    # 准备输入参数
    input_kwargs = {
        "hidden_states": latents,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": timesteps,
        "encoder_attention_mask": batch_data.get("encoder_attention_mask", None),
        "return_dict": False,
    }
    
    if "audio_embeds" in batch_data:
        input_kwargs["audio_emb"] = batch_data["audio_embeds"]
    if "face_embeds" in batch_data:
        input_kwargs["face_emb"] = batch_data["face_embeds"]
    
    if 'hunyuan' in model_type:
        input_kwargs["guidance"] = torch.tensor([1000.0], device=device, dtype=torch.bfloat16)
    
    # 前向传播
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model_pred = model(**input_kwargs)[0]
    
    # 分析激活值
    activation_info = {}
    for name, activation in activations.items():
        activation_info[name] = {
            "mean": activation.mean().item(),
            "std": activation.std().item(),
            "min": activation.min().item(),
            "max": activation.max().item()
        }
    
    # 清理钩子
    for hook in hooks:
        hook.remove()
    
    return activation_info

def visualize_results(anomalies, analysis_results, output_dir):
    """可视化分析结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制损失和梯度范数
    steps = [a["step"] for a in anomalies]
    losses = [a["loss"] for a in anomalies]
    grad_norms = [a["grad_norm"] for a in anomalies]
    analyzed_losses = [r["loss"] for r in analysis_results]
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(steps, losses, "bo-", label="训练时Loss")
    plt.plot(steps, analyzed_losses, "ro-", label="分析时Loss")
    plt.xlabel("步骤")
    plt.ylabel("Loss")
    plt.title("异常Loss对比")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(steps, grad_norms, "go-", label="梯度范数")
    plt.xlabel("步骤")
    plt.ylabel("梯度范数")
    plt.title("异常梯度范数")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "anomaly_overview.png"))
    plt.close()
    
    # 绘制每个异常的Top-10梯度分布
    for i, (anomaly, result) in enumerate(zip(anomalies, analysis_results)):
        top_grads = result["top_gradients"]
        names = list(top_grads.keys())[:10]
        norms = [top_grads[name]["norm"] for name in names]
        
        plt.figure(figsize=(12, 6))
        plt.barh(names, norms)
        plt.xlabel("梯度范数")
        plt.title(f"步骤 {anomaly['step']} 的Top-10梯度")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"top_grads_step_{anomaly['step']}.png"))
        plt.close()

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化设备
    device = torch.device(args.device)
    
    # 加载异常数据
    anomalies = load_anomaly_data(args.anomaly_dir, args.analyze_specific_step)
    print(f"加载了 {len(anomalies)} 条异常记录")
    
    if len(anomalies) == 0:
        print("没有找到异常记录，退出")
        return
    
    # 加载模型
    model = load_transformer(
        args.model_type,
        None,  # dit_model_path
        args.model_path,
        torch.bfloat16
    ).to(device)
    model.eval()  # 首先设置为评估模式
    
    # 分析每个异常
    analysis_results = []
    
    for anomaly in tqdm(anomalies, desc="分析异常"):
        # 加载批次数据
        batch_data = load_batch_data(anomaly, os.path.dirname(args.anomaly_dir))
        
        # 分析梯度
        model.train()  # 切换为训练模式以进行梯度分析
        gradient_info = analyze_gradients(model, batch_data, args.model_type, device)
        
        # 分析激活值 (如果实现了相关功能)
        model.eval()  # 切换为评估模式以分析激活值
        activation_info = analyze_activations(model, batch_data, args.model_type, device)
        
        # 保存分析结果
        result = {
            "step": anomaly["step"],
            "original_loss": anomaly["loss"],
            "original_grad_norm": anomaly["grad_norm"],
            "analysis_loss": gradient_info["loss"],
            "top_gradients": gradient_info["top_gradients"],
            "activation_info": activation_info
        }
        
        analysis_results.append(result)
        
        # 写入分析结果
        output_file = os.path.join(args.output_dir, f"analysis_step_{anomaly['step']}.json")
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"步骤 {anomaly['step']} 分析结果：")
        print(f"  原始 Loss: {anomaly['loss']:.4f}, 分析 Loss: {gradient_info['loss']:.4f}")
        print(f"  原始梯度范数: {anomaly['grad_norm']:.4f}")
        print("  Top-5 梯度模块:")
        for i, (name, info) in enumerate(list(gradient_info["top_gradients"].items())[:5]):
            print(f"    {i+1}. {name}: {info['norm']:.4f}")
        print()
    
    # 汇总分析
    summary = {
        "num_anomalies": len(anomalies),
        "avg_loss": np.mean([r["original_loss"] for r in analysis_results]),
        "avg_grad_norm": np.mean([r["original_grad_norm"] for r in analysis_results]),
        "avg_analysis_loss": np.mean([r["analysis_loss"] for r in analysis_results]),
    }
    
    with open(os.path.join(args.output_dir, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    # 可视化
    if args.visualize:
        visualize_results(anomalies, analysis_results, os.path.join(args.output_dir, "visualizations"))
        print(f"可视化结果已保存到 {os.path.join(args.output_dir, 'visualizations')}")

if __name__ == "__main__":
    main() 