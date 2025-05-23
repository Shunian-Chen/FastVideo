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

# 添加用于重建 mask 的常量
BACKGROUND_VALUE = 0.1
FACE_MASK_VALUE = 0.5
LIP_MASK_VALUE = 1.0

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
        # 修改 face_mask_dir 指向新的坐标文件目录
        self.face_mask_dir = os.path.join(self.datase_dir_path, "face_mask") # <-- 更新目录名
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
        # 获取 face mask 文件路径 (现在是坐标文件)
        face_mask_coord_file = self.data_anno[idx].get("face_emb_path") # 假设文件名与 face_emb 相同，只是在不同目录
        if face_mask_coord_file:
             # 兼容旧的 .png 后缀情况 (虽然新逻辑不应生成 png)
            if face_mask_coord_file.endswith(".png"):
                face_mask_coord_file = face_mask_coord_file.replace(".png", ".pt")
            elif not face_mask_coord_file.endswith(".pt"): # 确保是 .pt 文件
                 face_mask_coord_file += ".pt" # 或者采取其他修正方式

            face_mask_coord_path = os.path.join(self.face_mask_dir, face_mask_coord_file)
        else:
            face_mask_coord_path = None

        # load latent
        latent = torch.load(
            os.path.join(self.latent_dir, latent_file),
            map_location="cpu",
            weights_only=True,
        )
        latent = latent.squeeze(0)[:, -self.num_latent_t:]

        # load prompt
        if random.random() < self.cfg_rate:
            prompt_embed = self.uncond_prompt_embed
            prompt_attention_mask = self.uncond_prompt_mask
        else:
            prompt_embed = torch.load(
                os.path.join(self.prompt_embed_dir, prompt_embed_file),
                map_location="cpu",
                weights_only=True,
            )
            # Define path for potential error message
            prompt_attention_mask_path = os.path.join(self.prompt_attention_mask_dir, prompt_attention_mask_file)
            try:
                prompt_attention_mask = torch.load(
                    prompt_attention_mask_path,
                    map_location="cpu",
                    weights_only=True,
                )
            except Exception as e:
                logger.warning(f"Failed to load prompt attention mask {prompt_attention_mask_path}: {e}. Creating a fallback mask.")
                # Fallback: Create a boolean mask of ones with the same first dimension size as prompt_embed
                prompt_attention_mask = torch.ones(prompt_embed.shape[0]-1, dtype=torch.bool, device="cpu")
        

        # 加载 audio 和 face embeddings
        audio_embed = None
        face_embed = None
        if audio_embed_file:
            audio_embed_path = os.path.join(self.audio_embed_dir, audio_embed_file)
            if os.path.exists(audio_embed_path):
                audio_embed = torch.load(audio_embed_path, map_location="cpu", weights_only=False)
                if not isinstance(audio_embed, torch.Tensor): audio_embed = torch.as_tensor(audio_embed)
                audio_embed = self.process_audio_emb(audio_embed)
            else: logger.warning(f"Audio embed file not found: {audio_embed_path}")

        if face_embed_file:
            face_embed_path = os.path.join(self.face_embed_dir, face_embed_file)
            if os.path.exists(face_embed_path):
                face_embed = torch.load(face_embed_path, map_location="cpu", weights_only=False)
                if not isinstance(face_embed, torch.Tensor): face_embed = torch.as_tensor(face_embed)
            else: 
                face_embed = torch.ones(512)  # 创建512维全1张量
                # logger.warning(f"Face embed file not found: {face_embed_path}")


        # 加载和重建 face_mask
        face_mask = None
        if face_mask_coord_path and os.path.exists(face_mask_coord_path):
            try:
                # 加载坐标数据字典
                mask_data = torch.load(face_mask_coord_path, map_location="cpu")

                # 检查加载的数据格式
                if isinstance(mask_data, dict) and "original_height" in mask_data and "frames" in mask_data:
                    original_h = mask_data["original_height"]
                    original_w = mask_data["original_width"]
                    frame_coords_list = mask_data["frames"]
                    num_frames_in_file = len(frame_coords_list)

                    # 获取目标 latent 形状
                    num_latent_ch, num_latent_t_actual, target_h, target_w = latent.shape # 使用 latent 的实际 t 维度

                    # --- 重建 mask ---
                    reconstructed_masks = []
                    # 遍历坐标文件中的所有帧数据
                    for t_idx in range(num_frames_in_file): # <-- 使用 num_frames_in_file
                        # 创建单帧 mask (原始尺寸), 初始化为背景值
                        single_frame_mask = torch.full((original_h, original_w), BACKGROUND_VALUE, dtype=torch.float32)

                        # 确定从哪个坐标帧获取数据
                        # coord_idx = min(t_idx, num_frames_in_file - 1) if num_frames_in_file > 0 else -1 # No longer needed as we loop through num_frames_in_file
                        coord_idx = t_idx # Direct index

                        if coord_idx >= 0 and frame_coords_list[coord_idx] is not None:
                            coords = frame_coords_list[coord_idx]
                             # 确保 coords 包含8个值，否则认为是无效的
                            if len(coords) == 8:
                                fx1, fy1, fx2, fy2, lx1, ly1, lx2, ly2 = coords

                                # 填充面部区域 (确保坐标有效)
                                if fx2 > fx1 and fy2 > fy1:
                                    single_frame_mask[fy1:fy2, fx1:fx2] = FACE_MASK_VALUE

                                # 填充唇部区域 (确保坐标有效)
                                if lx2 > lx1 and ly2 > ly1:
                                    single_frame_mask[ly1:ly2, lx1:lx2] = LIP_MASK_VALUE
                            else:
                                logger.warning(f"Invalid coordinate format in frame {coord_idx} of {face_mask_coord_path}. Expected 8 values, got {len(coords)}. Using background mask for this frame.")
                        # else: 如果 coord_idx < 0 或 frame_coords_list[coord_idx] is None, mask 保持为背景值

                        reconstructed_masks.append(single_frame_mask)

                    if not reconstructed_masks:
                         logger.warning(f"未能重建任何 mask 帧 {face_mask_coord_path}. Using ones mask.")
                         face_mask = torch.ones(num_latent_ch, num_latent_t_actual, target_h, target_w) # Fallback
                    else:
                        face_mask = torch.stack(reconstructed_masks, dim=0) # Shape: [num_frames_in_file, H_orig, W_orig]
                        current_t, current_h, current_w = face_mask.shape[0], face_mask.shape[1], face_mask.shape[2]

                        # --- 新增：Center Crop 到 8x latent size ---
                        target_crop_h = target_h * 8
                        target_crop_w = target_w * 8

                        if current_h != target_crop_h or current_w != target_crop_w:
                            if current_h >= target_crop_h and current_w >= target_crop_w:
                                crop_y = (current_h - target_crop_h) // 2
                                crop_x = (current_w - target_crop_w) // 2
                                face_mask = face_mask[:, crop_y : crop_y + target_crop_h, crop_x : crop_x + target_crop_w] # Crop H, W dims
                                logger.debug(f"Cropped mask from {current_h, current_w} to {target_crop_h, target_crop_w}")
                                current_h, current_w = face_mask.shape[1], face_mask.shape[2] # Update current size after crop
                            else:
                                logger.warning(f"Original mask size ({current_h}, {current_w}) is smaller than target crop size ({target_crop_h, target_crop_w}). Skipping crop before interpolate.")
                        # --- 结束 Center Crop ---

                        # --- 调整尺寸和维度以匹配 latent (包括时间维度插值) ---
                        # 准备进行 3D 插值 (D, H, W) -> (T_latent, target_h, target_w)
                        # interpolate 需要 [N, C, D, H, W]
                        face_mask = face_mask.unsqueeze(0).unsqueeze(0) # [1, 1, T_file, H_crop, W_crop]

                        # 插值到目标尺寸 [T_latent, target_h, target_w]
                        face_mask = torch.nn.functional.interpolate(
                            face_mask,
                            size=(num_latent_t_actual, target_h, target_w), # Interpolate T, H, W
                            mode='trilinear', # Use trilinear for 3D interpolation
                            align_corners=False # Typically False for spatial dims, consider if True needed for time
                        ) # [1, 1, T_latent, target_h, target_w]

                        # 移除 N 和 C 维度
                        face_mask = face_mask.squeeze(0).squeeze(0) # [T_latent, target_h, target_w]

                        # 扩展/重复通道维度以匹配 latent [ch, t, target_h, target_w]
                        face_mask = face_mask.unsqueeze(0).repeat(num_latent_ch, 1, 1, 1) # [ch, T_latent, target_h, target_w]

                        # 保持原始 mask 值 (0.1, 0.5, 1.0) 以便后续可能的不同处理
                        # face_mask = (face_mask >= FACE_MASK_VALUE).float()


                elif isinstance(mask_data, torch.Tensor): # 兼容旧的 Tensor 格式 (.pt 文件)
                     logger.warning(f"Loaded face_mask is an old Tensor format: {face_mask_coord_path}. Applying direct interpolation (no 8x crop).")
                     face_mask = mask_data
                     if face_mask.dtype != torch.float32:
                         face_mask = face_mask.float()

                     # 预期维度是 [t, h, w]
                     if face_mask.ndim != 3:
                         logger.warning(f"Loaded .pt face_mask has unexpected dimensions: {face_mask.shape}. Expected [t, h, w]. Skipping resize.")
                         face_mask = torch.ones_like(latent) # Fallback
                     else:
                        # 获取目标 latent 形状
                         num_latent_ch, num_latent_t_actual, target_h, target_w = latent.shape
                         # 确保时间维度匹配 (裁剪或重复最后一帧)
                         if face_mask.shape[0] < num_latent_t_actual:
                             padding = torch.repeat_interleave(face_mask[-1:], num_latent_t_actual - face_mask.shape[0], dim=0)
                             face_mask = torch.cat([face_mask, padding], dim=0)
                         elif face_mask.shape[0] > num_latent_t_actual:
                             face_mask = face_mask[:num_latent_t_actual]

                         # 调整face_mask的尺寸以匹配latent的空间维度 [h, w]
                         # interpolate 需要 [N, C, H, W] 或 [N, C, D, H, W]
                         # 将 t 视为 N, 添加 C 维度
                         face_mask = face_mask.unsqueeze(1) # -> [t, 1, h, w]
                         face_mask = torch.nn.functional.interpolate(
                             face_mask,
                             size=(target_h, target_w),
                             mode='bilinear',
                             align_corners=False
                         ) # -> [t, 1, target_h, target_w]

                         face_mask = face_mask.squeeze(1) # -> [t, target_h, target_w]

                         # 扩展维度以匹配 latent [ch, t, h, w]
                         face_mask = face_mask.unsqueeze(0) # -> [1, t, target_h, target_w]
                         face_mask = face_mask.repeat(num_latent_ch, 1, 1, 1) # -> [ch, t, target_h, target_w]


                elif isinstance(mask_data, np.ndarray) or isinstance(mask_data, Image.Image): # 兼容旧的 png 文件 (如果用户手动移动了png到坐标目录)
                     logger.warning(f"Loaded face_mask appears to be an old image format (png?) moved to coord dir: {face_mask_coord_path}. Applying image processing logic.")
                     if isinstance(mask_data, Image.Image):
                         face_mask_np = np.array(mask_data)
                     else: # ndarray
                         face_mask_np = mask_data

                     face_mask = torch.from_numpy(face_mask_np)
                     # --- 从旧的png处理逻辑复制并调整 ---
                     # 获取目标 latent 形状
                     num_latent_ch, num_latent_t_actual, target_h, target_w = latent.shape
                     # 处理维度 HWC -> CHW or HW -> 1HW
                     if face_mask.ndim == 3:
                        face_mask = face_mask.permute(2, 0, 1)
                     elif face_mask.ndim == 2:
                        face_mask = face_mask.unsqueeze(0)
                     # 确保是单通道mask
                     if face_mask.shape[0] > 1:
                         logger.warning(f"Image mask has multiple channels ({face_mask.shape[0]}), using only the first channel.")
                         face_mask = face_mask[0:1, :, :]
                     elif face_mask.shape[0] == 0:
                         logger.error(f"Image mask has 0 channels. Fallback to ones.")
                         face_mask = torch.ones_like(latent)
                         # Skip further processing for this case
                         return latent, prompt_embed, prompt_attention_mask, audio_embed, face_embed, face_mask, audio_embed_file


                     # 转换为浮点数并归一化
                     if face_mask.dtype != torch.float32: face_mask = face_mask.float()
                     if face_mask.max() > 1.0: face_mask = face_mask / 255.0

                     # --- 新增：Center Crop 到 8x latent size ---
                     current_h, current_w = face_mask.shape[1], face_mask.shape[2]
                     target_crop_h = target_h * 8
                     target_crop_w = target_w * 8
                     if current_h != target_crop_h or current_w != target_crop_w:
                         if current_h >= target_crop_h and current_w >= target_crop_w:
                             crop_y = (current_h - target_crop_h) // 2
                             crop_x = (current_w - target_crop_w) // 2
                             face_mask = face_mask[:, crop_y : crop_y + target_crop_h, crop_x : crop_x + target_crop_w]
                             logger.debug(f"Cropped image mask from {current_h, current_w} to {target_crop_h, target_crop_w}")
                         else:
                             logger.warning(f"Image mask size ({current_h}, {current_w}) is smaller than target crop size ({target_crop_h}, {target_crop_w}). Skipping crop before interpolate.")
                     # --- 结束 Center Crop ---


                     # 调整尺寸 - Interpolate to latent size
                     # interpolate 需要 [N, C, H, W]
                     face_mask = torch.nn.functional.interpolate(
                         face_mask.unsqueeze(0), # [1, 1, H', W']
                         size=(target_h, target_w),
                         mode='bilinear', align_corners=False
                     ) # [1, 1, th, tw]
                     face_mask = face_mask.squeeze(0) # [1, th, tw]

                     # 重复时间维度
                     face_mask = face_mask.unsqueeze(1).repeat(num_latent_ch, num_latent_t_actual, 1, 1) # [ch, t, th, tw]


                else:
                    logger.warning(f"Loaded face_mask file format unknown or invalid: {face_mask_coord_path}. Using ones mask.")
                    face_mask = torch.ones_like(latent) # Fallback

            except Exception as e:
                logger.error(f"Error loading or processing face mask {face_mask_coord_path}: {e}", exc_info=True)
                face_mask = torch.ones_like(latent) # Fallback on error

        else:
            # logger.warning(f"Face mask coordinate file not found or not specified, using ones mask for latent {latent_file}")
            # 如果文件不存在或未在json中指定，使用全1掩码
            face_mask = torch.ones_like(latent)


        return latent, prompt_embed, prompt_attention_mask, audio_embed, face_embed, face_mask, audio_embed_file

    def __len__(self):
        return len(self.data_anno)


def latent_collate_function_audio(batch):
    latents, prompt_embeds, prompt_attention_masks, audio_embeds, face_embeds, face_masks, audio_embed_files = zip(*batch)

    # --- 处理 Latents ---
    # 计算 batch 中 latent 的最大尺寸
    max_t = max([latent.shape[1] for latent in latents])
    max_h = max([latent.shape[2] for latent in latents])
    max_w = max([latent.shape[3] for latent in latents])

    # Padding latents 到最大尺寸
    padded_latents = []
    original_latent_shapes = [] # 存储原始尺寸用于 attn mask
    for latent in latents:
        original_latent_shapes.append(latent.shape)
        ch, t_orig, h_orig, w_orig = latent.shape
        pad_t = max_t - t_orig
        pad_h = max_h - h_orig
        pad_w = max_w - w_orig
        padded_lat = torch.nn.functional.pad(latent, (0, pad_w, 0, pad_h, 0, pad_t))
        padded_latents.append(padded_lat)
    latents_stacked = torch.stack(padded_latents, dim=0)

    # --- 创建 Latent Attention Mask ---
    # 基于原始尺寸创建 mask
    latent_attn_mask = torch.zeros(len(latents), max_t, max_h, max_w) # 初始化为 0
    for i, shape in enumerate(original_latent_shapes):
        _, t_orig, h_orig, w_orig = shape
        latent_attn_mask[i, :t_orig, :h_orig, :w_orig] = 1 # 有效区域设为 1

    # --- 处理 Prompts ---
    prompt_embeds = torch.stack(prompt_embeds, dim=0)
    prompt_attention_masks = torch.stack(prompt_attention_masks, dim=0)

    # --- 处理 Face Masks ---
    # 在 __getitem__ 中 mask 已经与对应 latent 的（未padding）尺寸对齐
    # 需要将它们 padding 到 batch 内最大尺寸
    padded_face_masks = []
    are_masks_valid = all(isinstance(mask, torch.Tensor) for mask in face_masks) # 再次检查，以防万一

    if are_masks_valid:
        for i, face_mask in enumerate(face_masks):
            # face_mask shape is [ch, t_orig, h_orig, w_orig] matching original_latent_shapes[i]
            _, t_orig, h_orig, w_orig = original_latent_shapes[i]
            pad_t = max_t - t_orig
            pad_h = max_h - h_orig
            pad_w = max_w - w_orig

            # Padding value for mask should be 0 (or whatever value indicates "ignore")
            padded_mask = torch.nn.functional.pad(
                face_mask,
                (0, pad_w, 0, pad_h, 0, pad_t),
                mode='constant', value=0.0 # Pad mask with 0.0
            )
            padded_face_masks.append(padded_mask)
        face_masks_stacked = torch.stack(padded_face_masks, dim=0)
    else:
        logger.warning("Batch contains invalid face masks after __getitem__. Using ones mask for the batch.")
        face_masks_stacked = torch.ones_like(latents_stacked) # Fallback 全1掩码


    # --- 处理 Audio and Face Embeddings ---
    if audio_embeds[0] is not None:
        # 确保是 tensor 类型
        audio_embeds = [torch.as_tensor(emb) if not isinstance(emb, torch.Tensor) else emb for emb in audio_embeds]
        # 可能需要 padding audio embeds if lengths differ? Assuming fixed length or handled elsewhere for now.
        try:
             audio_embeds = torch.stack(audio_embeds, dim=0)
        except RuntimeError as e:
             logger.error(f"Error stacking audio embeddings: {e}. Check if audio embed dimensions are consistent across the batch.")
             # Handle inconsistent dimensions, e.g., pad or return None/error
             # For now, set to None if stacking fails
             audio_embeds = None
             logger.warning("Setting audio_embeds to None due to stacking error.")
    else:
        audio_embeds = None

    if face_embeds[0] is not None:
        # 确保是 tensor 类型
        face_embeds = [torch.as_tensor(emb) if not isinstance(emb, torch.Tensor) else emb for emb in face_embeds]
        try:
            face_embeds = torch.stack(face_embeds, dim=0)
        except RuntimeError as e:
             logger.error(f"Error stacking face embeddings: {e}. Check if face embed dimensions are consistent across the batch.")
             face_embeds = None
             logger.warning("Setting face_embeds to None due to stacking error.")

    else:
        face_embeds = None

    # 返回结果
    return latents_stacked, prompt_embeds, latent_attn_mask, prompt_attention_masks, audio_embeds, face_embeds, face_masks_stacked, audio_embed_files


if __name__ == "__main__":
    # 注意：确保此处的 json 文件引用的 face_emb_path 对应的文件是新的坐标格式 (.pt 字典)
    # 或者旧的 .pt tensor / .png 格式以测试兼容性
    dataset = LatentDatasetAudio(
        "/wangbenyou/shunian/workspace/talking_face/model_training/FastVideo/data/hallo3-data-origin-1k/videos2caption.json", # 替换为你的 json 路径
        num_latent_t=16, # 调整为你需要的 t 长度
        cfg_rate=0.1
        )

    # 更新 collate_fn
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=latent_collate_function_audio
        )

    # 更新迭代变量
    for batch_data in dataloader:
        # 解包 batch 数据
        latents, prompt_embeds, latent_attn_mask, prompt_attention_masks, audio_embeds, face_embeds, face_masks, audio_files = batch_data

        print("--- Batch Data ---")
        print(f"Latents shape: {latents.shape}, dtype: {latents.dtype}")
        print(f"Prompt Embeds shape: {prompt_embeds.shape}, dtype: {prompt_embeds.dtype}")
        print(f"Latent Attn Mask shape: {latent_attn_mask.shape}, dtype: {latent_attn_mask.dtype}, Unique values: {torch.unique(latent_attn_mask)}")
        print(f"Prompt Attn Mask shape: {prompt_attention_masks.shape}, dtype: {prompt_attention_masks.dtype}")
        print(f"Audio Embeds shape: {audio_embeds.shape if audio_embeds is not None else None}")
        print(f"Face Embeds shape: {face_embeds.shape if face_embeds is not None else None}")
        print(f"Face Masks shape: {face_masks.shape}, dtype: {face_masks.dtype}, Unique values: {torch.unique(face_masks)}")
        # print(f"Audio Files: {audio_files}") # 可能太长，注释掉

        # 可以添加断点进行更详细的检查
        # import pdb; pdb.set_trace()
        break # 只检查第一个 batch
