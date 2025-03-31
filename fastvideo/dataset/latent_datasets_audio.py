import json
import os
import random

import torch
from torch.utils.data import Dataset
from einops import rearrange
from loguru import logger
import sys
from PIL import Image
import numpy as np
## 不显示info级别的日志
logger.remove()
logger.add(sys.stdout, level="WARNING")

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
        concatenated_tensors = []

        for i in range(audio_emb.shape[0]):
            vectors_to_concat = [
                audio_emb[max(min(i + j, audio_emb.shape[0]-1), 0)]for j in range(-2, 3)]
            concatenated_tensors.append(torch.stack(vectors_to_concat, dim=0))

        audio_emb = torch.stack(concatenated_tensors, dim=0)

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
            # print(f"audio_embed.shape before processing: {audio_embed.shape}")
            audio_embed = self.process_audio_emb(audio_embed)
            # print(f"audio_embed.shape after processing: {audio_embed.shape}")
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
            if os.path.exists(os.path.join(self.face_mask_dir, face_mask_file)):
                face_mask = Image.open(os.path.join(self.face_mask_dir, face_mask_file))
                face_mask = torch.from_numpy(np.array(face_mask))
            else:
                face_mask = torch.ones(latent.shape[2], latent.shape[3])
                logger.warning(f"face_mask_file {face_mask_file} not found, using ones mask")
                # print(f"face_mask.shape: {face_mask.shape}")
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
            
            # print(f"face_mask.shape after processing: {face_mask.shape}")
            # print(f"latent.shape: {latent.shape}")
            # print(f"face mask type: {type(face_mask)}, dtype: {face_mask.dtype}")
            
            # 创建一个布尔掩码，用于标识非零元素（在计算loss时使用）
            face_mask_bool = (face_mask > 0.05).float()  # 转为0-1的浮点掩码，阈值可以调整
            # print(f"有效掩码区域占比: {face_mask_bool.sum() / face_mask_bool.numel():.4f}")
            
            # 使用布尔掩码代替原始掩码
            face_mask = face_mask_bool
            
        return latent, prompt_embed, prompt_attention_mask, audio_embed, face_embed, face_mask, audio_embed_file

    def __len__(self):
        return len(self.data_anno)


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
        # print(f"audio_embeds.shape before stacking: {audio_embeds[0].shape}")
        audio_embeds = torch.stack(audio_embeds, dim=0)
        # print(f"audio_embeds.shape after stacking: {audio_embeds.shape}")
    else:
        audio_embeds = None
        
    if face_embeds[0] is not None:
        # 确保是 tensor 类型
        face_embeds = [torch.as_tensor(emb) if not isinstance(emb, torch.Tensor) else emb for emb in face_embeds]
        face_embeds = torch.stack(face_embeds, dim=0)
    else:
        face_embeds = None

    return latents, prompt_embeds, latent_attn_mask, prompt_attention_masks, audio_embeds, face_embeds, face_masks, audio_embed_files


if __name__ == "__main__":
    dataset = LatentDatasetAudio("/wangbenyou/shunian/workspace/talking_face/model_training/FastVideo/data/hallo3-data-origin-1k/videos2caption.json",
                            num_latent_t=28,
                            cfg_rate=0.1)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=latent_collate_function_audio)
    for latent, prompt_embed, latent_attn_mask, prompt_attention_mask, audio_embed, face_embed in dataloader:
        print(
            latent.shape,
            prompt_embed.shape,
            latent_attn_mask.shape,
            prompt_attention_mask.shape,
            audio_embed.shape if audio_embed is not None else None,
            face_embed.shape if face_embed is not None else None,
        )
        import pdb

        pdb.set_trace()
