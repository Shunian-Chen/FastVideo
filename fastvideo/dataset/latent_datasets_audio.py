import json
import os
import random

import torch
from torch.utils.data import Dataset
from einops import rearrange
from loguru import logger
import sys
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
        if audio_embed_file:
            audio_embed = torch.load(
                os.path.join(self.audio_embed_dir, audio_embed_file),
                map_location="cpu",
                weights_only=False,
            )
            print(f"audio_embed from __getitem__ file: {audio_embed.shape}")
            print(f"audio_embed from __getitem__ file: {audio_embed_file}")
            # 确保是tensor类型
            if not isinstance(audio_embed, torch.Tensor):
                audio_embed = torch.as_tensor(audio_embed)
            print(f"audio_embed after as_tensor: {audio_embed.shape}")
            audio_embed = self.process_audio_emb(audio_embed)
            print(f"audio_embed after process_audio_emb: {audio_embed.shape}")
        if face_embed_file:
            face_embed = torch.load(
                os.path.join(self.face_embed_dir, face_embed_file),
                map_location="cpu",
                weights_only=False,
            )
            # 确保是tensor类型
            if not isinstance(face_embed, torch.Tensor):
                face_embed = torch.as_tensor(face_embed)
        return latent, prompt_embed, prompt_attention_mask, audio_embed, face_embed, audio_embed_file

    def __len__(self):
        return len(self.data_anno)


def latent_collate_function_audio(batch):
    # return latent, prompt, latent_attn_mask, text_attn_mask
    # latent_attn_mask: # b t h w
    # text_attn_mask: b 1 l
    # needs to check if the latent/prompt' size and apply padding & attn mask
    latents, prompt_embeds, prompt_attention_masks, audio_embeds, face_embeds, audio_embed_files = zip(*batch)
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

    return latents, prompt_embeds, latent_attn_mask, prompt_attention_masks, audio_embeds, face_embeds, audio_embed_files


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
