import os
import json
import torch
import logging
import argparse
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import glob
import torch.distributed as dist
from pathlib import Path
from torch.utils.data.distributed import DistributedSampler
from fastvideo.utils.parallel_states import (destroy_sequence_parallel_group,
                                             get_sequence_parallel_state,
                                             initialize_sequence_parallel_state
                                             )
from fastvideo.utils.communications import all_to_all
from PIL import Image
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 模拟nccl_info
class NCCLInfo:
    def __init__(self, sp_size=1):
        self.sp_size = sp_size

nccl_info = NCCLInfo()

def latent_collate_function_audio(batch):
    # return latent, prompt, latent_attn_mask, text_attn_mask
    # latent_attn_mask: # b t h w
    # text_attn_mask: b 1 l
    # needs to check if the latent/prompt' size and apply padding & attn mask
    latents, prompt_embeds, prompt_attention_masks, audio_embeds, face_embeds, face_masks, audio_embed_files = zip(*batch)
    # calculate max shape
    max_t = max([latent.shape[1] for latent in latents])
    max_h = max([latent.shape[2] for latent in latents])
    max_w = max([latent.shape[3] for latent in latents])

    # padding
    latents = [
        torch.nn.functional.pad(
            latent,
            (
                0,
                max_t - latent.shape[1],
                0,
                max_h - latent.shape[2],
                0,
                max_w - latent.shape[3],
            ),
        ) for latent in latents
    ]
    # attn mask
    latent_attn_mask = torch.ones(len(latents), max_t, max_h, max_w)
    # set to 0 if padding
    for i, latent in enumerate(latents):
        latent_attn_mask[i, latent.shape[1]:, :, :] = 0
        latent_attn_mask[i, :, latent.shape[2]:, :] = 0
        latent_attn_mask[i, :, :, latent.shape[3]:] = 0

    prompt_embeds = torch.stack(prompt_embeds, dim=0)
    prompt_attention_masks = torch.stack(prompt_attention_masks, dim=0)
    latents = torch.stack(latents, dim=0)
    
    # 处理face_masks的padding和stacking
    if face_masks[0] is not None:
        # 确保是tensor类型
        face_masks = [torch.as_tensor(mask) if not isinstance(mask, torch.Tensor) else mask for mask in face_masks]
        # 对于每个mask，确保维度匹配latent
        padded_face_masks = []
        for i, face_mask in enumerate(face_masks):
            # 现在face_mask的维度应该是[ch, t, h, w]，需要与latent完全一致
            # 首先检查通道维度
            if face_mask.shape[0] != latents[i].shape[0]:
                logger.warning(f"Face mask通道数 {face_mask.shape[0]} 与latent通道数 {latents[i].shape[0]} 不匹配")
                # 调整通道数以匹配latent
                face_mask = face_mask[:latents[i].shape[0]] if face_mask.shape[0] > latents[i].shape[0] else torch.cat([
                    face_mask, 
                    torch.zeros(latents[i].shape[0] - face_mask.shape[0], *face_mask.shape[1:], dtype=face_mask.dtype)
                ], dim=0)
            
            # 然后检查时间维度
            if face_mask.shape[1] != latents[i].shape[1]:
                if face_mask.shape[1] < latents[i].shape[1]:
                    # 需要在时间维度上padding
                    padding_size = latents[i].shape[1] - face_mask.shape[1]
                    face_mask = torch.nn.functional.pad(
                        face_mask,
                        (0, 0, 0, 0, 0, padding_size)  # padding in dim=1 (time dimension)
                    )
                else:
                    # 需要裁剪时间维度
                    face_mask = face_mask[:, :latents[i].shape[1]]
            
            # 检查高度和宽度
            if face_mask.shape[2:] != latents[i].shape[2:]:
                logger.warning(f"Face mask空间维度 {face_mask.shape[2:]} 与latent空间维度 {latents[i].shape[2:]} 不匹配")
                # 调整空间维度以匹配latent
                face_mask = torch.nn.functional.interpolate(
                    face_mask.view(face_mask.shape[0]*face_mask.shape[1], 1, *face_mask.shape[2:]),
                    size=latents[i].shape[2:],
                    mode='bilinear',
                    align_corners=False
                ).view(face_mask.shape[0], face_mask.shape[1], *latents[i].shape[2:])
            
            padded_face_masks.append(face_mask)
        
        face_masks = torch.stack(padded_face_masks, dim=0)
    else:
        # 创建一个全1的掩码，表示所有区域都参与loss计算
        face_masks = torch.ones_like(latents)

    # 处理audio和face embeddings
    if audio_embeds[0] is not None:
        # 确保是 tensor 类型
        audio_embeds = [torch.as_tensor(emb) if not isinstance(emb, torch.Tensor) else emb for emb in audio_embeds]
        audio_embeds = torch.stack(audio_embeds, dim=0)
    else:
        audio_embeds = None
        
    if face_embeds[0] is not None:
        # 确保是 tensor 类型
        face_embeds = [torch.as_tensor(emb) if not isinstance(emb, torch.Tensor) else emb for emb in face_embeds]
        face_embeds = torch.stack(face_embeds, dim=0)
    else:
        face_embeds = None

    return latents, prompt_embeds, latent_attn_mask, prompt_attention_masks, audio_embeds, face_embeds, face_masks, audio_embed_files

# 模拟prepare_sequence_parallel_data_audio函数
def prepare_sequence_parallel_data_audio(hidden_states, encoder_hidden_states,
                                   attention_mask, encoder_attention_mask, 
                                   audio_emb, face_emb, face_mask):
    if nccl_info.sp_size == 1:
        return (
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            encoder_attention_mask,
            audio_emb,
            face_emb,
            face_mask,
        )

    def prepare(hidden_states, encoder_hidden_states, attention_mask,
                encoder_attention_mask, audio_emb, face_emb, face_mask):
        hidden_states = all_to_all(hidden_states, scatter_dim=2, gather_dim=0)
        encoder_hidden_states = all_to_all(encoder_hidden_states,
                                           scatter_dim=1,
                                           gather_dim=0)
        attention_mask = all_to_all(attention_mask,
                                    scatter_dim=1,
                                    gather_dim=0)
        encoder_attention_mask = all_to_all(encoder_attention_mask,
                                            scatter_dim=1,
                                            gather_dim=0)
        audio_emb = all_to_all(audio_emb,
                              scatter_dim=1,
                              gather_dim=0)
        face_emb = all_to_all(face_emb,
                              scatter_dim=1,
                              gather_dim=0)
        face_mask = all_to_all(face_mask,
                              scatter_dim=2,
                              gather_dim=0)
        
        return (
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            encoder_attention_mask,
            audio_emb,
            face_emb,
            face_mask,
        )

    sp_size = nccl_info.sp_size
    frame = hidden_states.shape[2]
    # 如果frame不是sp_size的倍数，则需要进行padding，hidden_states的维度是(bs, frame, h, w, d)
    if frame % sp_size != 0:
        remainder = frame % sp_size
        padding = sp_size - remainder
        pad_shape = list(hidden_states.shape)
        pad_shape[2] = padding
        zeros_pad = torch.zeros(pad_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = torch.cat([hidden_states, zeros_pad], dim=2)
    
    frame = hidden_states.shape[2]
    assert frame % sp_size == 0, "frame should be a multiple of sp_size"

    # 如果audio_frame不是sp_size的倍数，则需要进行padding
    audio_frame = audio_emb.shape[1]
    if audio_frame % nccl_info.sp_size != 0:
        remainder = audio_frame % nccl_info.sp_size
        padding = nccl_info.sp_size - remainder
        # 构造需要补齐的零张量
        pad_shape = list(audio_emb.shape)
        pad_shape[1] = padding
        zeros_pad = torch.zeros(pad_shape, dtype=audio_emb.dtype, device=audio_emb.device)
        # 拼接到原张量上完成padding
        audio_emb = torch.cat([audio_emb, zeros_pad], dim=1)

    (
        hidden_states,
        encoder_hidden_states,
        attention_mask,
        encoder_attention_mask,
        audio_emb,
        face_emb,
        face_mask,
    ) = prepare(
        hidden_states,
        encoder_hidden_states.repeat(1, sp_size, 1),
        attention_mask.repeat(1, sp_size, 1, 1),
        encoder_attention_mask.repeat(1, sp_size),
        audio_emb,
        face_emb.repeat(1, sp_size),
        face_mask,
    )


    return hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask, audio_emb, face_emb, face_mask

# 模拟sp_parallel_dataloader_wrapper_audio函数
def sp_parallel_dataloader_wrapper_audio(dataloader, device, train_batch_size,
                                   sp_size, train_sp_batch_size):
    while True:
        for data_item in dataloader:
            latents, cond, attn_mask, cond_mask, audio_emb, face_emb, face_mask, audio_embed_file = data_item
            latents = latents.to(device)
            cond = cond.to(device)
            attn_mask = attn_mask.to(device)
            cond_mask = cond_mask.to(device)
            # 确保audio_emb和face_emb是tensor类型并移到正确设备
            if audio_emb is not None:
                if not isinstance(audio_emb, torch.Tensor):
                    audio_emb = torch.as_tensor(audio_emb)
                audio_emb = audio_emb.to(device)
            if face_emb is not None:
                if not isinstance(face_emb, torch.Tensor):
                    face_emb = torch.as_tensor(face_emb)
                face_emb = face_emb.to(device)
            if face_mask is not None:
                if not isinstance(face_mask, torch.Tensor):
                    face_mask = torch.as_tensor(face_mask)
                face_mask = face_mask.to(device)
            frame = latents.shape[2]
            if frame == 1:
                yield latents, cond, attn_mask, cond_mask, audio_emb, face_emb, face_mask, audio_embed_file
            else:
                latents, cond, attn_mask, cond_mask, audio_emb, face_emb, face_mask = prepare_sequence_parallel_data_audio(
                    latents, cond, attn_mask, cond_mask, audio_emb, face_emb, face_mask)
                
                assert (
                    train_batch_size * sp_size >= train_sp_batch_size
                ), "train_batch_size * sp_size should be greater than train_sp_batch_size"
                for iter in range(train_batch_size * sp_size //
                                  train_sp_batch_size):
                    st_idx = iter * train_sp_batch_size
                    ed_idx = (iter + 1) * train_sp_batch_size
                    encoder_hidden_states = cond[st_idx:ed_idx]
                    encoder_attention_mask = cond_mask[st_idx:ed_idx]
                    audio_emb_batch = audio_emb[st_idx:ed_idx]
                    face_emb_batch = face_emb[st_idx:ed_idx]
                    face_mask_batch = face_mask[st_idx:ed_idx]

                    # 检查是否有空的audio_emb
                    if audio_emb_batch.shape[0] == 0:
                        logger.error(f"{'*' * 80}")
                        logger.error(f"audio_embeds_file['{audio_embed_file}'] is empty")
                        logger.error(f"{'*' * 80}")
                    
                    yield (
                        latents[st_idx:ed_idx],
                        encoder_hidden_states,
                        encoder_attention_mask,
                        audio_emb_batch,
                        face_emb_batch,
                        face_mask_batch,
                        audio_embed_file,
                    )

# 真实数据集类
class LatentDatasetAudio(Dataset):

    def __init__(
        self,
        json_path,
        num_latent_t,
        cfg_rate,
    ):
        # data_merge_path: video_dir, latent_dir, prompt_embed_dir, json_path
        self.json_path = json_path
        self.cfg_rate = cfg_rate
        self.datase_dir_path = os.path.dirname(json_path)
        self.video_dir = os.path.join(self.datase_dir_path, "video")
        self.latent_dir = os.path.join(self.datase_dir_path, "latent")
        self.face_mask_dir = os.path.join(self.datase_dir_path, "face_mask")
        self.prompt_embed_dir = os.path.join(self.datase_dir_path,
                                             "prompt_embed")
        self.prompt_attention_mask_dir = os.path.join(self.datase_dir_path,
                                                      "prompt_attention_mask")
        self.audio_embed_dir = os.path.join(self.datase_dir_path, "audio_emb")
        self.face_embed_dir = os.path.join(self.datase_dir_path, "face_emb")
        with open(self.json_path, "r") as f:
            self.data_anno = json.load(f)
        # json.load(f) already keeps the order
        # self.data_anno = sorted(self.data_anno, key=lambda x: x['latent_path'])
        self.num_latent_t = num_latent_t
        # just zero embeddings [256, 4096]
        self.uncond_prompt_embed = torch.zeros(256, 4096).to(torch.float32)
        # 256 zeros
        self.uncond_prompt_mask = torch.zeros(256).bool()
        self.lengths = [
            data_item["length"] if "length" in data_item else 1
            for data_item in self.data_anno
        ]

    def process_audio_emb(self, audio_emb):
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

    def __getitem__(self, idx):
        latent_file = self.data_anno[idx]["latent_path"]
        prompt_embed_file = self.data_anno[idx]["prompt_embed_path"]
        prompt_attention_mask_file = self.data_anno[idx][
            "prompt_attention_mask"]
        # 获取audio和face embedding文件路径
        audio_embed_file = self.data_anno[idx].get("audio_emb_path")
        face_embed_file = self.data_anno[idx].get("face_emb_path")
        face_mask_file = self.data_anno[idx].get("face_emb_path").replace("pt", "png")
        # load
        latent = torch.load(
            os.path.join(self.latent_dir, latent_file),
            map_location="cpu",
            weights_only=True,
        )
        latent = latent.squeeze(0)[:, -self.num_latent_t:]
        print(f"latent.shape: {latent.shape}")
        if random.random() < self.cfg_rate:
            prompt_embed = self.uncond_prompt_embed
            prompt_attention_mask = self.uncond_prompt_mask
        else:
            prompt_embed = torch.load(
                os.path.join(self.prompt_embed_dir, prompt_embed_file),
                map_location="cpu",
                weights_only=True,
            )
            prompt_attention_mask = torch.load(
                os.path.join(self.prompt_attention_mask_dir,
                             prompt_attention_mask_file),
                map_location="cpu",
                weights_only=True,
            )

        # 加载audio和face embeddings
        audio_embed = None
        face_embed = None
        face_mask = None
        if audio_embed_file:
            audio_embed = torch.load(
                os.path.join(self.audio_embed_dir, audio_embed_file),
                map_location="cpu",
                weights_only=False,
            )
            # 确保是tensor类型
            if not isinstance(audio_embed, torch.Tensor):
                audio_embed = torch.as_tensor(audio_embed)
            audio_embed = self.process_audio_emb(audio_embed)
        if face_embed_file:
            face_embed = torch.load(
                os.path.join(self.face_embed_dir, face_embed_file),
                map_location="cpu",
                weights_only=False,
            )
            # 确保是tensor类型
            if not isinstance(face_embed, torch.Tensor):
                face_embed = torch.as_tensor(face_embed)
        if face_mask_file:
            face_mask = Image.open(os.path.join(self.face_mask_dir, face_mask_file))
            face_mask = torch.from_numpy(np.array(face_mask))
            print(f"face_mask.shape: {face_mask.shape}")
            # 需要将face mask的维度与latent对齐，其中，latent的维度是[16, 13, 60, 106], 分别对应[ch, t, h, w], 而face mask的维度是[480, 848], 分别对应[h, w]
            
            # 首先将face_mask转换为浮点数类型并归一化到0-1范围
            if face_mask.dtype != torch.float32:
                face_mask = face_mask.float()
                # 如果是0-255范围，归一化到0-1
                if face_mask.max() > 1.0:
                    face_mask = face_mask / 255.0
            
            # 调整face_mask的尺寸以匹配latent的空间维度
            face_mask = torch.nn.functional.interpolate(
                face_mask.unsqueeze(0).unsqueeze(0),  # 增加批次和通道维度 [1, 1, h, w]
                size=(latent.shape[2], latent.shape[3]),  # 目标尺寸为latent的高度和宽度
                mode='bilinear',
                align_corners=False
            )
            
            # 在时间维度上重复以匹配latent的时间维度
            face_mask = face_mask.repeat(1, latent.shape[1], 1, 1)  # [1, t, h, w]
            
            # 为便于与latent直接相乘计算masked loss，再在通道维度上进行扩展
            # 扩展通道维度，从[1, t, h, w]变为[1, t, h, w, 1]
            face_mask = face_mask.unsqueeze(-1)
            # 在通道维度上重复，得到与latent第一维相同的通道数
            face_mask = face_mask.repeat(1, 1, 1, 1, latent.shape[0])
            # 调整维度顺序，变为[1, ch, t, h, w]
            face_mask = face_mask.permute(0, 4, 1, 2, 3)
            # 去掉批次维度，变为[ch, t, h, w]
            face_mask = face_mask.squeeze(0)
            
            print(f"face_mask.shape after processing: {face_mask.shape}")
            print(f"latent.shape: {latent.shape}")
            print(f"face mask type: {type(face_mask)}, dtype: {face_mask.dtype}")
            
            # 创建一个布尔掩码，用于标识非零元素（在计算loss时使用）
            face_mask_bool = (face_mask > 0.05).float()  # 转为0-1的浮点掩码，阈值可以调整
            print(f"有效掩码区域占比: {face_mask_bool.sum() / face_mask_bool.numel():.4f}")
            
            # 使用布尔掩码代替原始掩码
            face_mask = face_mask_bool
            
        return latent, prompt_embed, prompt_attention_mask, audio_embed, face_embed, face_mask, audio_embed_file

    def __len__(self):
        return len(self.data_anno)

# 模拟train_one_step函数
def train_one_step(loader, iterations=5):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    for i in range(iterations):
        try:
            (
                latents,
                encoder_hidden_states,
                encoder_attention_mask,
                audio_embeds,
                face_embeds,
                face_mask,
                audio_embed_file,
            ) = next(loader)
            if rank == 0:
                logger.info(f"Iteration {i+1}/{iterations}")
                logger.info(f"Loaded data successfully:")
                logger.info(f"  latents: {latents.shape}")
                logger.info(f"  audio_embeds: {audio_embeds.shape}")
                logger.info(f"  face_embeds: {face_embeds.shape}")
                logger.info(f"  face_mask: {face_mask.shape}")
                
            # 检查是否有空的audio_embeds
            if audio_embeds.shape[0] == 0:
                logger.error(f"{'*' * 80}")
                logger.error(f"audio_embeds_file['{audio_embed_file}'] is empty")
                logger.error(f"{'*' * 80}")
                # 这里可以添加处理空audio_embeds的逻辑
        
        except Exception as e:
            logger.error(f"Error in iteration {i+1}: {e}")


def test_with_real_data(data_dir, sp_size=2, batch_size=2, sp_batch_size=1, iterations=5):
    """
    使用真实数据测试
    
    参数:
        data_dir: 数据目录
        sp_size: 序列并行大小
        batch_size: 批次大小
        sp_batch_size: 序列并行批次大小
        iterations: 迭代次数
    """
    # 设置序列并行大小
    global nccl_info
    nccl_info = NCCLInfo(sp_size=sp_size)
    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(sp_size)

    # 创建真实数据集和数据加载器
    # logger.info(f"加载真实数据集，目录: {data_dir}")
    dataset = LatentDatasetAudio(data_dir, num_latent_t=32, cfg_rate=0.0)
    
    sampler = DistributedSampler(
        dataset, rank=rank, num_replicas=world_size, shuffle=False)

    train_dataloader = DataLoader(
        dataset,
        sampler=sampler,
        collate_fn=latent_collate_function_audio,
        pin_memory=True,
        batch_size=batch_size,
        num_workers=0,
        drop_last=True,
    )
    
    # 创建包装后的数据加载器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 测试原始实现
    # logger.info("测试原始实现...")
    original_loader = sp_parallel_dataloader_wrapper_audio(
        train_dataloader, 
        device, 
        batch_size,
        sp_size,
        sp_batch_size
    )
    
    try:
        train_one_step(original_loader, iterations=iterations)
    except Exception as e:
        logger.error(f"原始实现失败: {e}")
    

# 添加一个示例函数，演示如何使用face_mask计算masked loss
def calculate_masked_loss(pred, target, face_mask):
    """
    计算带有面部掩码的损失，只考虑face_mask不为零的部分
    
    参数:
        pred: 预测结果，形状为[batch, ch, t, h, w]
        target: 目标值，形状为[batch, ch, t, h, w]
        face_mask: 面部掩码，形状为[batch, ch, t, h, w]，值范围在0-1之间
        
    返回:
        masked_loss: 只计算掩码不为零部分的损失
    """
    # 确保掩码形状与预测结果匹配
    assert pred.shape == face_mask.shape, f"预测形状 {pred.shape} 与掩码形状 {face_mask.shape} 不匹配"
    
    # 计算每个元素的损失（例如使用MSE）
    per_element_loss = torch.nn.functional.mse_loss(pred, target, reduction='none')
    
    # 应用掩码，只计算掩码非零部分的损失
    masked_loss = per_element_loss * face_mask
    
    # 计算平均损失（只考虑掩码非零的元素）
    # 首先计算掩码非零元素的总数
    non_zero_elements = face_mask.sum()
    
    if non_zero_elements > 0:
        # 如果有非零元素，则只对这些元素计算平均损失
        masked_loss_sum = masked_loss.sum()
        masked_loss_avg = masked_loss_sum / non_zero_elements
    else:
        # 如果掩码全为零，则返回零损失
        masked_loss_avg = torch.tensor(0.0, device=pred.device)
    
    return masked_loss_avg

# 示例：如何在训练循环中使用
def example_training_step(model, latents, face_mask, optimizer):
    """
    训练步骤示例，演示如何在实际训练中使用face_mask
    """
    # 确保所有输入在同一设备上
    device = latents.device
    face_mask = face_mask.to(device)
    
    # 前向传播
    pred = model(latents)
    
    # 假设我们有真实的目标值
    target = latents  # 这里简化为自重建任务
    
    # 计算带掩码的损失
    loss = calculate_masked_loss(pred, target, face_mask)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def main():
    parser = argparse.ArgumentParser(description="Test audio loader")
    parser.add_argument("--sp_size", type=int, default=2, help="Sequence parallel size")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--sp_batch_size", type=int, default=1, help="Sequence parallel batch size")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations to test")
    parser.add_argument("--test_fix", action="store_true", help="Test the fixed implementation")
    parser.add_argument("--use_real_data", action="store_true", help="Use real data for testing")
    parser.add_argument("--data_dir", type=str, default="", help="Directory containing real data")
    args = parser.parse_args()
    

    test_with_real_data(
        args.data_dir,
        sp_size=args.sp_size,
        batch_size=args.batch_size,
        sp_batch_size=args.sp_batch_size,
        iterations=args.iterations
    )



if __name__ == "__main__":
    main() 

'''
torchrun --nnodes 1 --nproc_per_node 4 test_audio_loader.py --use_real_data --data_dir data/hallo3-data-origin-1k/videos2caption.json --sp_size 4 --batch_size 1 --sp_batch_size 1
'''