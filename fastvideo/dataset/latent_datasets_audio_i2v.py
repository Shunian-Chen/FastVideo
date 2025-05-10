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
from fastvideo.models.hunyuan.constants import PROMPT_TEMPLATE, PRECISION_TO_TYPE
import torchvision.transforms as transforms
from typing import List
import PIL
from fastvideo.models.hunyuan.constants import BACKGROUND_VALUE, FACE_MASK_VALUE, LIP_MASK_VALUE


## 不显示info级别的日志
logger.remove()
logger.add(sys.stdout, level="WARNING")


def numpy_to_pil(images: np.ndarray) -> List[PIL.Image.Image]:
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def black_image(width, height):
    black_image = Image.new("RGB", (width, height), (0, 0, 0))
    return black_image

def get_cond_images(args, latents, vae, is_uncond=False):
    """get conditioned images by decode the first frame latents"""
    if len(latents.shape) == 5:
        sematic_image_latents = latents[:, :, 0, ...]
        sematic_image_latents = sematic_image_latents.unsqueeze(2).to(vae.dtype)
    elif len(latents.shape) == 4:
        sematic_image_latents = latents[:, 0, ...]
        sematic_image_latents = sematic_image_latents.unsqueeze(1).to(vae.dtype)
        sematic_image_latents = sematic_image_latents.unsqueeze(0).to(vae.dtype)

    else: 
        sematic_image_latents = latents
    sematic_image_latents = 1 / vae.config.scaling_factor * sematic_image_latents.to(vae.device)

    # print(f"sematic_image_latents device: {sematic_image_latents.device}")
    # print(f"vae device: {vae.device}")
    semantic_images = vae.decode(
        sematic_image_latents, return_dict=False
    )[0]

    semantic_images = semantic_images.squeeze(2)
    semantic_images = (semantic_images / 2 + 0.5).clamp(0, 1)
    semantic_images = semantic_images.cpu().permute(0, 2, 3, 1).float().numpy()
    # print(f"semantic image shape: {semantic_images.shape}")

    semantic_images = numpy_to_pil(semantic_images)
    if is_uncond:
        semantic_images = [
            black_image(img.size[0], img.size[1]) for img in semantic_images
        ]

    # 将semantic_images以png的格式保存到指定路径
    for i, img in enumerate(semantic_images):
        img.save(os.path.join("/data/nas/yexin/workspace/shunian/model_training/FastVideo/outputs_video", f"semantic_images_{i}.png"))
    return semantic_images

def get_cond_latents(args, latents, vae):
    """get conditioned latent by decode and encode the first frame latents"""
    if len(latents.shape) == 5:
        first_image_latents = latents[:, :, 0, ...]  
        first_image_latents = first_image_latents.unsqueeze(2).to(vae.dtype)
    elif len(latents.shape) == 4:
        first_image_latents = latents[:, 0, ...]  
        first_image_latents = first_image_latents.unsqueeze(1).to(vae.dtype)
        first_image_latents = first_image_latents.unsqueeze(0).to(vae.dtype)

    else: 
        first_image_latents = latents
    
    # breakpoint()
    # print(f"first_image_latents device before: {first_image_latents.device}")
    # print(f"vae device before: {vae.device}")
    first_image_latents = 1 / vae.config.scaling_factor * first_image_latents.to(vae.device)

    # print(f"latents shape: {latents.shape}, dtype: {latents.dtype}")
    # print(f"first image latents shape: {first_image_latents.shape}, dtype: {first_image_latents.dtype}")
    # print(f"first image latents shape after unsqueeze: {first_image_latents.unsqueeze(1).shape}, dtype: {first_image_latents.unsqueeze(2).dtype}")

    # print(f"first_image_latents device: {first_image_latents.device}")
    # print(f"vae device: {vae.device}")
    first_images = vae.decode(
        first_image_latents, return_dict=False
    )[0]

    if len(first_images.shape) == 5:
        first_images = first_images.squeeze(2)
    elif len(first_images.shape) == 4:
        first_images = first_images.squeeze(1)
    
    # print(f"first image shape: {first_images.shape}, dtype: {first_images.dtype}")
    first_images = (first_images / 2 + 0.5).clamp(0, 1)
    first_images = first_images.cpu().permute(0, 2, 3, 1).float().numpy()
    first_images = numpy_to_pil(first_images)

    # 将first_images以png的格式保存到指定路径
    for i, img in enumerate(first_images):
        img.save(os.path.join("/data/nas/yexin/workspace/shunian/model_training/FastVideo/outputs_video", f"first_images_{i}.png"))

    image_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )
    first_images_pixel_values = [image_transform(image) for image in first_images]
    first_images_pixel_values = (
        torch.cat(first_images_pixel_values).unsqueeze(0).unsqueeze(2).to(vae.device)
    )
    # print(f"first_images_pixel_values shape: {first_images_pixel_values.shape}, dtype: {first_images_pixel_values.dtype}")
    vae_dtype = PRECISION_TO_TYPE[args.vae_precision]
    with torch.autocast(
        device_type="cuda", dtype=vae_dtype, enabled=vae_dtype != torch.float32
    ):
        cond_latents = vae.encode(
            first_images_pixel_values
        ).latent_dist.sample()  # B, C, F, H, W
        cond_latents.mul_(vae.config.scaling_factor)

    # print(f"cond_latents shape: {cond_latents.shape}, dtype: {cond_latents.dtype}")
    return cond_latents


def get_text_tokens(text_encoder, description):
    text_inputs = text_encoder.text2tokens(description, data_type='video')
    text_ids = text_inputs["input_ids"]
    text_mask = text_inputs["attention_mask"]
    return text_ids, text_mask

def get_text_hidden_states(text_encoder, text_ids, text_mask, semantic_images):
    text_outputs = text_encoder.encode(
            {"input_ids": text_ids, "attention_mask": text_mask},
            data_type="video",
            semantic_images=semantic_images,
        )
    text_states = text_outputs.hidden_state
    text_mask = text_outputs.attention_mask
    return text_states, text_mask

class LatentDatasetAudio_i2v(Dataset):

    def __init__(
        self,
        json_path,
        num_latent_t,
    ):
        # data_merge_path: video_dir, latent_dir, prompt_embed_dir, json_path
        self.json_path = json_path
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
        # print("latent_file",latent_file)
        # prompt_embed_file = self.data_anno[idx]["prompt_embed_path"]
        # prompt_attention_mask_file = self.data_anno[idx][
        #     "prompt_attention_mask"]
        # 获取audio和face embedding文件路径
        sample_id = self.data_anno[idx]["latent_path"].split('.')[0]
        audio_embed_file = self.data_anno[idx].get("audio_emb_path")
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
        # cond_latents = get_cond_latents(self.args, latent, self.vae)
        
        caption = self.data_anno[idx]["caption"]
        # print(f"caption in data anno: {caption}")

        # 加载 audio 和 face embeddings
        audio_embed = None
        if audio_embed_file:
            audio_embed_path = os.path.join(self.audio_embed_dir, audio_embed_file)
            if os.path.exists(audio_embed_path):
                audio_embed = torch.load(audio_embed_path, map_location="cpu", weights_only=False)
                if not isinstance(audio_embed, torch.Tensor): audio_embed = torch.as_tensor(audio_embed)
                audio_embed = self.process_audio_emb(audio_embed)
            else: logger.warning(f"Audio embed file not found: {audio_embed_path}")

        # 加载和重建 face_mask
        face_mask = None
        if face_mask_coord_path and os.path.exists(face_mask_coord_path):
            try:
                # 加载坐标数据字典
                mask_data = torch.load(face_mask_coord_path, map_location="cpu", weights_only=True)

                # 检查加载的数据格式
                # if isinstance(mask_data, dict) and "original_height" in mask_data and "frames" in mask_data:
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
                        logger.error(f"未能重建任何 mask 帧 {face_mask_coord_path}. Using ones mask.")
                        face_mask = torch.ones(num_latent_ch, num_latent_t_actual, target_h, target_w) # Fallback
                else:
                    face_mask = torch.stack(reconstructed_masks, dim=0) # Shape: [num_frames_in_file, H_orig, W_orig]
                    current_t, current_h, current_w = face_mask.shape[0], face_mask.shape[1], face_mask.shape[2]

                    # --- New Aspect-Ratio Preserving Center Crop (inspired by load_and_process_face_mask) ---
                    # target_h, target_w are latent H, W (e.g., 32x32 or 64x64)
                    # Current_h, current_w are original H, W of face_mask (e.g., 1024x1024 or source video resolution)

                    # Target dimensions for the aspect ratio of the crop region are 8x the latent dimensions
                    ar_crop_target_h = target_h * 8
                    ar_crop_target_w = target_w * 8
                    
                    face_mask_after_ar_crop = face_mask # Default to current face_mask if AR crop is not possible or not needed

                    if ar_crop_target_h <= 0 or ar_crop_target_w <= 0:
                        logger.warning(f"Target AR crop dimensions ({ar_crop_target_h}, {ar_crop_target_w}) are invalid for {sample_id}. Using un-AR-cropped mask ({current_h}x{current_w}).")
                    elif current_h <= 0 or current_w <=0: 
                        logger.warning(f"Original mask dimensions for AR cropping ({current_h}, {current_w}) are invalid for {sample_id}. Using un-AR-cropped mask.")
                    else:
                        # Calculate aspect ratios
                        # target_ar_for_crop is the AR of the 8*latent_dim region we want to achieve
                        target_ar_for_crop = float(ar_crop_target_h) / ar_crop_target_w
                        # input_ar_current_mask is AR of current face_mask (e.g. 1024x1024 mask)
                        input_ar_current_mask = float(current_h) / current_w

                        calc_crop_h_pixels = 0
                        calc_crop_w_pixels = 0

                        if input_ar_current_mask > target_ar_for_crop:
                            # Mask is 'taller' or less 'wide' than the target AR. Crop height.
                            calc_crop_h_pixels = int(current_w * target_ar_for_crop) # New height based on current_w and target AR
                            calc_crop_w_pixels = current_w                           # Width remains current_w
                        else:
                            # Mask is 'wider' or less 'tall' than the target AR (or same AR). Crop width.
                            calc_crop_h_pixels = current_h                           # Height remains current_h
                            calc_crop_w_pixels = int(current_h / target_ar_for_crop) # New width based on current_h and target AR
                        
                        if calc_crop_h_pixels > 0 and calc_crop_w_pixels > 0:
                            crop_y_start_pixels = int(round((current_h - calc_crop_h_pixels) / 2.0))
                            crop_x_start_pixels = int(round((current_w - calc_crop_w_pixels) / 2.0))
                            
                            crop_y_start_pixels = max(0, crop_y_start_pixels)
                            crop_x_start_pixels = max(0, crop_x_start_pixels)

                            # Final actual crop dimensions, ensuring they don't go out of bounds
                            actual_crop_h_pixels = min(calc_crop_h_pixels, current_h - crop_y_start_pixels)
                            actual_crop_w_pixels = min(calc_crop_w_pixels, current_w - crop_x_start_pixels)

                            if actual_crop_h_pixels > 0 and actual_crop_w_pixels > 0:
                                face_mask_after_ar_crop = face_mask[
                                    :, # All frames
                                    crop_y_start_pixels : crop_y_start_pixels + actual_crop_h_pixels,
                                    crop_x_start_pixels : crop_x_start_pixels + actual_crop_w_pixels
                                ]
                                logger.debug(f"Aspect-ratio preserved cropped mask for {sample_id} from ({current_h},{current_w}) to ({actual_crop_h_pixels},{actual_crop_w_pixels}) targeting AR ~{target_ar_for_crop:.2f} (for 8*latent region).")
                            else:
                                logger.warning(f"Final actual AR crop dimensions ({actual_crop_h_pixels}, {actual_crop_w_pixels}) are invalid for {sample_id}. Using un-AR-cropped mask ({current_h}x{current_w}).")
                        else:
                            logger.warning(f"Initial calculated AR crop dimensions ({calc_crop_h_pixels}, {calc_crop_w_pixels}) are invalid for {sample_id}. Using un-AR-cropped mask ({current_h}x{current_w}).")
                    
                    face_mask = face_mask_after_ar_crop # Assign the (potentially) AR-cropped mask back
                    # Update current_h and current_w to reflect the dimensions of `face_mask` *after* this AR crop,
                    # as these will be the input dimensions for the subsequent interpolation step.
                    current_h, current_w = face_mask.shape[1], face_mask.shape[2] 
                    # --- End of New Aspect-Ratio Preserving Center Crop ---

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


                # elif isinstance(mask_data, torch.Tensor): # 兼容旧的 Tensor 格式 (.pt 文件)
                #      logger.warning(f"Loaded face_mask is an old Tensor format: {face_mask_coord_path}. Applying direct interpolation (no 8x crop).")
                #      face_mask = mask_data
                #      if face_mask.dtype != torch.float32:
                #          face_mask = face_mask.float()

                #      # 预期维度是 [t, h, w]
                #      if face_mask.ndim != 3:
                #          logger.warning(f"Loaded .pt face_mask has unexpected dimensions: {face_mask.shape}. Expected [t, h, w]. Skipping resize.")
                #          face_mask = torch.ones_like(latent) # Fallback
                #      else:
                #         # 获取目标 latent 形状
                #          num_latent_ch, num_latent_t_actual, target_h, target_w = latent.shape
                #          # 确保时间维度匹配 (裁剪或重复最后一帧)
                #          if face_mask.shape[0] < num_latent_t_actual:
                #              padding = torch.repeat_interleave(face_mask[-1:], num_latent_t_actual - face_mask.shape[0], dim=0)
                #              face_mask = torch.cat([face_mask, padding], dim=0)
                #          elif face_mask.shape[0] > num_latent_t_actual:
                #              face_mask = face_mask[:num_latent_t_actual]

                #          # 调整face_mask的尺寸以匹配latent的空间维度 [h, w]
                #          # interpolate 需要 [N, C, H, W] 或 [N, C, D, H, W]
                #          # 将 t 视为 N, 添加 C 维度
                #          face_mask = face_mask.unsqueeze(1) # -> [t, 1, h, w]
                #          face_mask = torch.nn.functional.interpolate(
                #              face_mask,
                #              size=(target_h, target_w),
                #              mode='bilinear',
                #              align_corners=False
                #          ) # -> [t, 1, target_h, target_w]

                #          face_mask = face_mask.squeeze(1) # -> [t, target_h, target_w]

                #          # 扩展维度以匹配 latent [ch, t, h, w]
                #          face_mask = face_mask.unsqueeze(0) # -> [1, t, target_h, target_w]
                #          face_mask = face_mask.repeat(num_latent_ch, 1, 1, 1) # -> [ch, t, target_h, target_w]


                # elif isinstance(mask_data, np.ndarray) or isinstance(mask_data, Image.Image): # 兼容旧的 png 文件 (如果用户手动移动了png到坐标目录)
                #      logger.warning(f"Loaded face_mask appears to be an old image format (png?) moved to coord dir: {face_mask_coord_path}. Applying image processing logic.")
                #      if isinstance(mask_data, Image.Image):
                #          face_mask_np = np.array(mask_data)
                #      else: # ndarray
                #          face_mask_np = mask_data

                #      face_mask = torch.from_numpy(face_mask_np)
                #      # --- 从旧的png处理逻辑复制并调整 ---
                #      # 获取目标 latent 形状
                #      num_latent_ch, num_latent_t_actual, target_h, target_w = latent.shape
                #      # 处理维度 HWC -> CHW or HW -> 1HW
                #      if face_mask.ndim == 3:
                #         face_mask = face_mask.permute(2, 0, 1)
                #      elif face_mask.ndim == 2:
                #         face_mask = face_mask.unsqueeze(0)
                #      # 确保是单通道mask
                #      if face_mask.shape[0] > 1:
                #          logger.warning(f"Image mask has multiple channels ({face_mask.shape[0]}), using only the first channel.")
                #          face_mask = face_mask[0:1, :, :]
                #      elif face_mask.shape[0] == 0:
                #          logger.error(f"Image mask has 0 channels. Fallback to ones.")
                #          face_mask = torch.ones_like(latent)
                #          # Skip further processing for this case
                #          return latent, caption, audio_embed, face_mask


                #      # 转换为浮点数并归一化
                #      if face_mask.dtype != torch.float32: face_mask = face_mask.float()
                #      if face_mask.max() > 1.0: face_mask = face_mask / 255.0

                #      # --- 新增：Center Crop 到 8x latent size ---
                #      current_h, current_w = face_mask.shape[1], face_mask.shape[2]
                #      target_crop_h = target_h * 8
                #      target_crop_w = target_w * 8
                #      if current_h != target_crop_h or current_w != target_crop_w:
                #          if current_h >= target_crop_h and current_w >= target_crop_w:
                #              crop_y = (current_h - target_crop_h) // 2
                #              crop_x = (current_w - target_crop_w) // 2
                #              face_mask = face_mask[:, crop_y : crop_y + target_crop_h, crop_x : crop_x + target_crop_w]
                #              logger.debug(f"Cropped image mask from {current_h, current_w} to {target_crop_h, target_crop_w}")
                #          else:
                #              logger.warning(f"Image mask size ({current_h}, {current_w}) is smaller than target crop size ({target_crop_h}, {target_crop_w}). Skipping crop before interpolate.")
                #      # --- 结束 Center Crop ---


                #      # 调整尺寸 - Interpolate to latent size
                #      # interpolate 需要 [N, C, H, W]
                #      face_mask = torch.nn.functional.interpolate(
                #          face_mask.unsqueeze(0), # [1, 1, H', W']
                #          size=(target_h, target_w),
                #          mode='bilinear', align_corners=False
                #      ) # [1, 1, th, tw]
                #      face_mask = face_mask.squeeze(0) # [1, th, tw]

                #      # 重复时间维度
                #      face_mask = face_mask.unsqueeze(1).repeat(num_latent_ch, num_latent_t_actual, 1, 1) # [ch, t, th, tw]


                # else:
                #     logger.error(f"Loaded face_mask file format unknown or invalid: {face_mask_coord_path}. Using ones mask.")
                #     face_mask = torch.ones_like(latent) # Fallback

            except Exception as e:
                logger.error(f"Error loading or processing face mask {face_mask_coord_path}: {e}", exc_info=True)
                face_mask = torch.ones_like(latent) # Fallback on error

        else:
            logger.error(f"Face mask coordinate file not found or not specified, using ones mask for latent {latent_file}")
            # 如果文件不存在或未在json中指定，使用全1掩码
            face_mask = torch.ones_like(latent)


        # print(f"face_mask shape: {face_mask.shape}")
        # print(f"face_mask dtype: {face_mask.dtype}")
        # # print(f"face_mask unique values: {torch.unique(face_mask)}")
        # print(f"caption: {caption}")
        # print(f"audio_embed shape: {audio_embed.shape}")
        # print(f"latent shape: {latent.shape}")

        return latent, caption, audio_embed, face_mask, sample_id

    def __len__(self):
        return len(self.data_anno)


def latent_collate_function_audio_i2v(batch):
    latents, captions, audio_embeds, face_masks, sample_ids = zip(*batch)

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

    # # --- 处理 Prompts ---
    # prompt_embeds = torch.stack(prompt_embeds, dim=0)
    # prompt_attention_masks = torch.stack(prompt_attention_masks, dim=0)

    # --- 处理 Cond Latents ---
    # print(f"cond_latents shape before stack: {len(cond_latents)}, dtype: {cond_latents[0].dtype}")
    # cond_latents = torch.stack(cond_latents, dim=0)
    # print(f"cond_latents shape after stack: {cond_latents.shape}, dtype: {cond_latents.dtype}")

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
    if audio_embeds is not None and audio_embeds[0] is not None:
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

    # if face_embeds[0] is not None:
    #     # 确保是 tensor 类型
    #     face_embeds = [torch.as_tensor(emb) if not isinstance(emb, torch.Tensor) else emb for emb in face_embeds]
    #     try:
    #         face_embeds = torch.stack(face_embeds, dim=0)
    #     except RuntimeError as e:
    #          logger.error(f"Error stacking face embeddings: {e}. Check if face embed dimensions are consistent across the batch.")
    #          face_embeds = None
    #          logger.warning("Setting face_embeds to None due to stacking error.")

    # else:
    #     face_embeds = None

    # 返回结果
    return latents_stacked, captions, audio_embeds, face_masks_stacked, sample_ids


if __name__ == "__main__":
    # 注意：确保此处的 json 文件引用的 face_emb_path 对应的文件是新的坐标格式 (.pt 字典)
    # 或者旧的 .pt tensor / .png 格式以测试兼容性
    # import torch.multiprocessing as mp


    # os.environ['MODEL_BASE'] = "/data/nas/yexin/workspace/shunian/model"
    class Args:
        # 从 finetune_hunyuan_audio_i2v.sh 和 train_audio_i2v.py 推断的参数
        vae_precision = "bf16"
        text_encoder = "llm-i2v"
        text_encoder_precision = "fp16"
        text_len = 256
        tokenizer = "llm-i2v"
        prompt_template = "dit-llm-encode-i2v"
        prompt_template_video = "dit-llm-encode-video-i2v"
        hidden_state_skip_layer = 2
        i2v_mode = True
        reproduce = True
        apply_final_norm = False  # train_audio_i2v.py 中的默认值
        model_type = "hunyuan_audio_i2v" # 从 finetune 脚本推断
        num_latent_t = 32 # 从 finetune 脚本推断
        # sematic_cond_drop_p 不是 argparse 参数，为测试设置一个默认值
        sematic_cond_drop_p = 0.1
        # cfg 在 dataset 中用于随机 uncond，在 train 中用于 guidance scale，这里保留示例值
        cfg = 0 # --cfg in train script is 0.0, but dataset uses it differently

    args = Args() # 实例化 Args

    # if args.i2v_mode:
    #     image_embed_interleave = 4
    # else:
    #     image_embed_interleave = 1


    # from fastvideo.utils.load import load_vae
    # # 使用 finetune 脚本中的 VAE 路径和 args 中的 model_type
    # vae, _, _ = load_vae(model_type=args.model_type, pretrained_model_name_or_path="/data/nas/yexin/workspace/shunian/model/")
    # vae = vae.to("cuda")

    # from fastvideo.models.hunyuan.text_encoder import TextEncoder_i2v
    # text_encoder = TextEncoder_i2v(
    #         text_encoder_type=args.text_encoder,
    #         max_length=args.text_len
    #         + (
    #             PROMPT_TEMPLATE[args.prompt_template_video].get("crop_start", 0)
    #             if args.prompt_template_video is not None
    #             else PROMPT_TEMPLATE[args.prompt_template].get("crop_start", 0)
    #             if args.prompt_template is not None
    #             else 0
    #         ),
    #         text_encoder_precision=args.text_encoder_precision,
    #         tokenizer_type=args.tokenizer,
    #         i2v_mode=args.i2v_mode,
    #         prompt_template=(
    #             PROMPT_TEMPLATE[args.prompt_template]
    #             if args.prompt_template is not None
    #             else None
    #         ),
    #         prompt_template_video=(
    #             PROMPT_TEMPLATE[args.prompt_template_video]
    #             if args.prompt_template_video is not None
    #             else None
    #         ),
    #         hidden_state_skip_layer=args.hidden_state_skip_layer,
    #         apply_final_norm=args.apply_final_norm,
    #         reproduce=args.reproduce,
    #         logger=logger,
    #         device="cuda",
    #         image_embed_interleave=image_embed_interleave
    #     )

    dataset = LatentDatasetAudio_i2v(
        "/data/nas/yexin/workspace/shunian/model_training/FastVideo/data/252_hour_test_480p_49frames/videos2caption.json", # 使用 finetune 脚本中的数据路径
        num_latent_t=args.num_latent_t, # 使用 args 中的值
        cfg_rate=args.cfg, # 使用 args 中的值
        )
    # 更新 collate_fn
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=latent_collate_function_audio_i2v
        )

    # 更新迭代变量
    for batch_data in dataloader:
        # 解包 batch 数据
        latents, captions, audio_embeds, face_masks, sample_ids = batch_data

        print("--- Batch Data ---")
        print(f"Latents shape: {latents.shape}, dtype: {latents.dtype}")
        print(f"Captions : {captions}")
        print(f"Audio Embeds shape: {audio_embeds.shape if audio_embeds is not None else None}")
        print(f"Face Masks shape: {face_masks.shape}, dtype: {face_masks.dtype}, Unique values: {torch.unique(face_masks)}")

        # 可以添加断点进行更详细的检查
        # import pdb; pdb.set_trace()
        break # 只检查第一个 batch
