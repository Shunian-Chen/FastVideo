# !/bin/python3
# isort: skip_file
import argparse
import math
import os
import sys
import time
import json
import datetime
from collections import deque
import numpy as np
import torch.multiprocessing as mp
import torch
import torch.distributed as dist
import wandb
from accelerate.utils import set_seed
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, convert_unet_state_dict_to_peft
from peft import LoraConfig, set_peft_model_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
import PIL
import random
from typing import List
from torch.cuda.amp import GradScaler
from fastvideo.dataset.latent_datasets import (LatentDataset,
                                               latent_collate_function)
from fastvideo.dataset.latent_datasets_audio import (LatentDatasetAudio,
                                                     latent_collate_function_audio)
from fastvideo.dataset.latent_datasets_audio_i2v import (LatentDatasetAudio_i2v,
                                                     latent_collate_function_audio_i2v,
                                                     get_cond_latents,
                                                     get_cond_images)
from fastvideo.models.mochi_hf.mochi_latents_utils import normalize_dit_input
from fastvideo.models.mochi_hf.pipeline_mochi import MochiPipeline
from fastvideo.models.hunyuan_hf.pipeline_hunyuan import HunyuanVideoPipeline

from fastvideo.utils.checkpoint import (resume_lora_optimizer, save_checkpoint,
                                        save_lora_checkpoint, resume_checkpoint)
from fastvideo.utils.communications import (broadcast,
                                            sp_parallel_dataloader_wrapper,
                                            sp_parallel_dataloader_wrapper_audio,
                                            sp_parallel_dataloader_wrapper_audio_i2v)
from fastvideo.utils.dataset_utils import LengthGroupedSampler
from fastvideo.utils.fsdp_util import (apply_fsdp_checkpointing,
                                       get_dit_fsdp_kwargs)
from fastvideo.utils.load import load_transformer
from fastvideo.utils.logging_ import main_print
from fastvideo.utils.parallel_states import (destroy_sequence_parallel_group,
                                             get_sequence_parallel_state,
                                             initialize_sequence_parallel_state
                                             )
from fastvideo.utils.validation import log_validation
from loguru import logger
from fastvideo.utils.load import load_vae

from fastvideo.models.hunyuan.constants import *
from fastvideo.utils.load import load_text_encoder
from fastvideo.dataset.latent_datasets_audio_i2v import BACKGROUND_VALUE, FACE_MASK_VALUE, LIP_MASK_VALUE

import sys
############################# new #############################
from transformers import get_cosine_schedule_with_warmup
############################# new #############################

## 不显示info级别的日志
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger.remove()
logger.add(sys.stdout, level="WARNING")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    generator,
    logit_mean: float = None,
    logit_std: float = None,
    mode_scale: float = None,
):
    """
    Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(
            mean=logit_mean,
            std=logit_std,
            size=(batch_size, ),
            device="cpu",
            generator=generator,
        )
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size, ), device="cpu", generator=generator)
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2)**2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size, ), device="cpu", generator=generator)
    return u


def get_sigmas(noise_scheduler,
               device,
               timesteps,
               n_dim=4,
               dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item()
                    for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def _prepare_batch_data(args, batch_data, vae, text_encoder, model_type):
    """Prepare batch data including latents, conditioning, and prompts."""
    latents, captions, audio_embeds, face_mask, sample_ids = batch_data
    latents.mul_(vae.config.scaling_factor)

    cond_latents = get_cond_latents(args, latents, vae)

    is_uncond = (
        torch.tensor(1).to(torch.int64)
        if random.random() < args.sematic_cond_drop_p
        else torch.tensor(0).to(torch.int64)
    )
    semantic_images = get_cond_images(args, latents, vae, is_uncond=is_uncond)

    if random.random() < args.cfg:
        prompt_embed = torch.zeros(257, 4096, dtype=torch.float32, device=latents.device)
        prompt_attention_mask = torch.zeros(256, dtype=torch.bool, device=latents.device)
        prompt_embed = prompt_embed.repeat(latents.shape[0], 1, 1)
        prompt_attention_mask = prompt_attention_mask.repeat(latents.shape[0], 1)
    else:
        prompt_embed, prompt_attention_mask = text_encoder.encode_prompt(
            captions, semantic_images=semantic_images
        )

    latents = normalize_dit_input(model_type, latents)

    return (
        latents, audio_embeds, face_mask, sample_ids, cond_latents,
        semantic_images, prompt_embed, prompt_attention_mask, captions
    )


def _add_noise_and_schedule(
    latents, noise_scheduler, noise_random_generator, weighting_scheme,
    logit_mean, logit_std, mode_scale, sp_size, device
):
    """Add noise to latents based on sampled timesteps."""
    batch_size = latents.shape[0]
    noise = torch.randn_like(latents)
    u = compute_density_for_timestep_sampling(
        weighting_scheme=weighting_scheme,
        batch_size=batch_size,
        generator=noise_random_generator,
        logit_mean=logit_mean,
        logit_std=logit_std,
        mode_scale=mode_scale,
    )
    indices = (u * noise_scheduler.config.num_train_timesteps).long()
    timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)
    if sp_size > 1:
        broadcast(timesteps)

    sigmas = get_sigmas(
        noise_scheduler, latents.device, timesteps, n_dim=latents.ndim, dtype=latents.dtype
    )
    noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
    return noisy_model_input, noise, timesteps, sigmas


def _model_forward(
    transformer, model_type, noisy_model_input, prompt_embed, timesteps,
    prompt_attention_mask, audio_embeds, face_mask
):
    """Perform the forward pass of the transformer model."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        input_kwargs = {
            "hidden_states": noisy_model_input,
            "encoder_hidden_states": prompt_embed,
            "timestep": timesteps,
            "encoder_attention_mask": prompt_attention_mask,
            "return_dict": False,
            "audio_emb": audio_embeds if audio_embeds is not None else None,
            "face_mask": face_mask
        }
        if 'hunyuan' in model_type:
            input_kwargs["guidance"] = torch.tensor(
                [1000.0], device=noisy_model_input.device, dtype=torch.bfloat16
            )
        model_pred = transformer(**input_kwargs)[0]
    return model_pred


def _calculate_target(precondition_outputs, model_type, cond_latents, latents, noise):
    """Calculate the target tensor for loss computation."""
    if precondition_outputs:
        if "audio_i2v" in model_type:
            target = torch.concat([cond_latents, latents[:, :, 1:, :, :]], dim=2)
        else:
            target = latents
    else:
        if "audio_i2v" in model_type:
            target = torch.concat([cond_latents, (noise - latents)[:, :, 1:, :, :]], dim=2)
        else:
            target = noise - latents
    return target


def _calculate_and_accumulate_losses(
    model_pred, target, face_mask, gradient_accumulation_steps,
    accum_bg_loss, accum_face_loss, accum_lip_loss,
    accum_bg_count, accum_face_count, accum_lip_count,
    epsilon=0.5
):
    """Calculate loss, regional losses, and accumulate them."""
    # --- Calculate Squared Error ---
    squared_error = (model_pred.float() - target.float())**2

    # --- Calculate Spatial Loss Map (Average over c, t) ---
    spatial_loss_map = torch.mean(squared_error, dim=(1, 2)) # Shape: [b, h, w]

    # --- Calculate Weighted Spatial Loss Map (Average over c, t after weighting) ---
    weighted_squared_error = squared_error * face_mask # Shape: [b, c, t, h, w]
    weighted_spatial_loss_map = torch.mean(weighted_squared_error, dim=(1, 2)) # Shape: [b, h, w]

    # --- Identify Regions using Epsilon ---
    background_mask = (torch.abs(face_mask - BACKGROUND_VALUE) < epsilon).float()
    face_region_mask = (torch.abs(face_mask - FACE_MASK_VALUE) < epsilon).float()
    lip_region_mask = (torch.abs(face_mask - LIP_MASK_VALUE) < epsilon).float()

    # --- Calculate and Accumulate Batch Regional Losses ---
    current_bg_count = torch.sum(background_mask)
    current_face_count = torch.sum(face_region_mask)
    current_lip_count = torch.sum(lip_region_mask)

    current_bg_loss = torch.sum(squared_error * background_mask)
    current_face_loss = torch.sum(squared_error * face_region_mask)
    current_lip_loss = torch.sum(squared_error * lip_region_mask)

    accum_bg_loss += current_bg_loss
    accum_face_loss += current_face_loss
    accum_lip_loss += current_lip_loss
    accum_bg_count += current_bg_count
    accum_face_count += current_face_count
    accum_lip_count += current_lip_count

    # --- Calculate Overall Loss for Backpropagation (Original Method) ---
    loss = (torch.sum(squared_error * face_mask) /
            (torch.sum(face_mask) + 1e-8)) / gradient_accumulation_steps + 1e-8

    return (
        loss, squared_error, spatial_loss_map, weighted_spatial_loss_map,
        background_mask, face_region_mask, lip_region_mask,
        accum_bg_loss, accum_face_loss, accum_lip_loss,
        accum_bg_count, accum_face_count, accum_lip_count
    )


def _save_debug_info(
    is_nan, nan_debug_dir, sample_loss_dir, rank, step, micro_batch_idx,
    data_for_nan_debug, loss_value, squared_error, background_mask,
    face_region_mask, lip_region_mask, spatial_loss_map, weighted_spatial_loss_map,
    sample_ids, logger
):
    """Save debug information, including NaN data, sample losses, and heatmaps."""
    batch_size = squared_error.shape[0] # Get batch size from one of the tensors

    if is_nan and nan_debug_dir:
        debug_prefix = f"nan_debug_rank_{rank}_step_{step}_mb_{micro_batch_idx+1}"
        metadata = {}
        try:
            for key, value in data_for_nan_debug.items():
                if isinstance(value, torch.Tensor):
                    save_path = os.path.join(nan_debug_dir, f"{debug_prefix}_{key}.pt")
                    torch.save(value.cpu().detach(), save_path) # Ensure saving CPU tensors
                    metadata[key] = save_path
                elif isinstance(value[0], np.ndarray) or isinstance(value[0], PIL.Image.Image):
                    save_path = os.path.join(nan_debug_dir, f"{debug_prefix}_{key}.pt")
                    torch.save(value[0].cpu().detach(), save_path) # Ensure saving CPU tensors
                    metadata[key] = save_path
                elif value is not None:
                    metadata[key] = value

            metadata["calculated_loss_before_nan_check"] = loss_value.item() # The actual NaN value
            torch.save(squared_error.cpu().detach(), os.path.join(nan_debug_dir, f"{debug_prefix}_squared_error.pt"))
            torch.save(background_mask.cpu().detach(), os.path.join(nan_debug_dir, f"{debug_prefix}_background_mask_calc.pt"))
            torch.save(face_region_mask.cpu().detach(), os.path.join(nan_debug_dir, f"{debug_prefix}_face_region_mask_calc.pt"))
            torch.save(lip_region_mask.cpu().detach(), os.path.join(nan_debug_dir, f"{debug_prefix}_lip_region_mask_calc.pt"))

            metadata_path = os.path.join(nan_debug_dir, f"{debug_prefix}_metadata.json")
            with open(metadata_path, 'w') as f:
                def safe_serialize(obj):
                    if isinstance(obj, torch.Tensor):
                        return f"Tensor shape: {obj.shape}, dtype: {obj.dtype}"
                    elif isinstance(obj, np.ndarray):
                        return f"Numpy array shape: {obj.shape}, dtype: {obj.dtype}"
                    elif isinstance(obj, PIL.Image.Image):
                        return f"PIL Image size: {obj.size}, mode: {obj.mode}"
                    try:
                        json.dumps(obj)
                        return obj
                    except (TypeError, OverflowError):
                        return str(obj)

                sanitized_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, list):
                        sanitized_metadata[key] = [safe_serialize(item) for item in value]
                    elif isinstance(value, dict):
                        sanitized_metadata[key] = {k: safe_serialize(v) for k, v in value.items()}
                    else:
                        sanitized_metadata[key] = safe_serialize(value)
                json.dump(sanitized_metadata, f, indent=4)
            logger.info(f"Rank {rank}: Successfully saved NaN debug data with prefix {debug_prefix}")
        except Exception as e:
            logger.error(f"Rank {rank}: Failed to save NaN debug data for step {step}, micro-batch {micro_batch_idx+1}. Error: {e}")

    if sample_loss_dir:
        with torch.no_grad():
            for b in range(batch_size):
                sample_id = sample_ids[b]
                sample_sq_error = squared_error[b]
                sample_bg_mask = background_mask[b]
                sample_face_mask = face_region_mask[b]
                sample_lip_mask = lip_region_mask[b]

                sample_bg_loss = torch.sum(sample_sq_error * sample_bg_mask) / (torch.sum(sample_bg_mask) + 1e-8)
                sample_face_loss = torch.sum(sample_sq_error * sample_face_mask) / (torch.sum(sample_face_mask) + 1e-8)
                sample_lip_loss = torch.sum(sample_sq_error * sample_lip_mask) / (torch.sum(sample_lip_mask) + 1e-8)

                loss_data = {
                    "sample_id": sample_id,
                    "background_loss": sample_bg_loss.item() if torch.sum(sample_bg_mask) > 0 else 0.0,
                    "face_loss": sample_face_loss.item() if torch.sum(sample_face_mask) > 0 else 0.0,
                    "lip_loss": sample_lip_loss.item() if torch.sum(sample_lip_mask) > 0 else 0.0,
                    "step": step,
                }

                json_filename = f"{sample_id}_rank_{rank}_step_{step}_mb_{micro_batch_idx+1}.json"
                output_path = os.path.join(sample_loss_dir, json_filename)
                try:
                    with open(output_path, 'w') as f:
                        json.dump(loss_data, f, indent=4)
                except Exception as e:
                    logger.error(f"Rank {rank}: Failed to save sample loss for {sample_id} at step {step}, micro-batch {micro_batch_idx+1} to {output_path}: {e}")

                sample_spatial_loss = spatial_loss_map[b].detach().cpu()
                heatmap_filename = f"{sample_id}_rank_{rank}_step_{step}_mb_{micro_batch_idx+1}_heatmap.pt"
                heatmap_output_path = os.path.join(sample_loss_dir, heatmap_filename)
                try:
                    torch.save(sample_spatial_loss, heatmap_output_path)
                except Exception as e:
                    logger.error(f"Rank {rank}: Failed to save spatial loss heatmap for {sample_id} at step {step}, micro-batch {micro_batch_idx+1} to {heatmap_output_path}: {e}")

                sample_weighted_spatial_loss = weighted_spatial_loss_map[b].detach().cpu()
                weighted_heatmap_filename = f"{sample_id}_rank_{rank}_step_{step}_mb_{micro_batch_idx+1}_weighted_loss_heatmap.pt"
                weighted_heatmap_output_path = os.path.join(sample_loss_dir, weighted_heatmap_filename)
                try:
                    torch.save(sample_weighted_spatial_loss, weighted_heatmap_output_path)
                except Exception as e:
                    logger.error(f"Rank {rank}: Failed to save weighted spatial loss heatmap for {sample_id} at step {step}, micro-batch {micro_batch_idx+1} to {weighted_heatmap_output_path}: {e}")


def _aggregate_and_compute_final_losses(
    accum_bg_loss, accum_face_loss, accum_lip_loss,
    accum_bg_count, accum_face_count, accum_lip_count
):
    """Aggregate regional losses across ranks and compute final averages."""
    dist.all_reduce(accum_bg_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(accum_face_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(accum_lip_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(accum_bg_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(accum_face_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(accum_lip_count, op=dist.ReduceOp.SUM)

    avg_bg_loss_batch = (accum_bg_loss / (accum_bg_count + 1e-8)).item() if accum_bg_count > 0 else 0.0
    avg_face_loss_batch = (accum_face_loss / (accum_face_count + 1e-8)).item() if accum_face_count > 0 else 0.0
    avg_lip_loss_batch = (accum_lip_loss / (accum_lip_count + 1e-8)).item() if accum_lip_count > 0 else 0.0

    return avg_bg_loss_batch, avg_face_loss_batch, avg_lip_loss_batch


def _optimizer_step_and_lr_update(
    transformer, optimizer, lr_scheduler, max_grad_norm, logger, rank, step
):
    """Perform gradient clipping, optimizer step, and LR scheduler step."""
    grad_norm = None
    clipping_error = False
    try:
        grad_norm = transformer.clip_grad_norm_(max_grad_norm)
        grad_norm = grad_norm.item()
    except Exception as e:
        logger.error(f"Rank {rank}: Error during gradient clipping at step {step}: {e}. Skipping optimizer step.")
        clipping_error = True
        optimizer.zero_grad() # Zero grads even if clipping failed

    # 2️⃣ 检查范数是否 finite（注意 torch.isfinite 支持 float / tensor）
    if not torch.isfinite(torch.tensor(grad_norm)):
        logger.error(
            f"Rank {rank}: Non‑finite grad_norm ({grad_norm}) detected at step {step}. "
            "Skipping optimizer update."
        )
        optimizer.zero_grad(set_to_none=True)
        lr_scheduler.step()   # 可选：也可以选择不更新 LR
        return float('nan'), True   # 第二个返回值表示“本 step 发生错误”
    
    if not clipping_error:
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    return grad_norm, clipping_error


def train_one_step(
    transformer,
    model_type,
    optimizer,
    lr_scheduler,
    loader,
    noise_scheduler,
    noise_random_generator,
    gradient_accumulation_steps,
    sp_size,
    precondition_outputs,
    max_grad_norm,
    weighting_scheme,
    logit_mean,
    logit_std,
    mode_scale,
    vae,
    text_encoder,
    args,
    device,
    logger,
    rank,
    output_dir,
    step,
    nan_debug_dir,
    scaler,
):
    total_loss = 0.0
    optimizer.zero_grad() # Zero gradients at the beginning of the step

    # Define epsilon for region matching
    epsilon = 0.5
    sample_loss_dir = os.path.join(output_dir, "sample_losses") if output_dir else None

    # Initialize accumulators for batch average regional losses
    accum_bg_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    accum_face_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    accum_lip_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    accum_bg_count = torch.tensor(0.0, device=device, dtype=torch.float32)
    accum_face_count = torch.tensor(0.0, device=device, dtype=torch.float32)
    accum_lip_count = torch.tensor(0.0, device=device, dtype=torch.float32)

    nan_detected_in_step = False # Flag to track NaNs within the step
    successful_micro_batches = 0 # Count successful micro-batches for averaging loss

    # Accumulate gradients
    for i in range(gradient_accumulation_steps):
        try:
            batch_data = next(loader)
        except Exception as e:
             logger.error(f"Rank {rank}: Error getting next batch from dataloader: {e}")
             continue # Skip this micro-batch

        # --- 1. Prepare Batch Data ---
        (latents, audio_embeds, face_mask, sample_ids, cond_latents,
         semantic_images, prompt_embed, prompt_attention_mask, captions) = _prepare_batch_data(
            args, batch_data, vae, text_encoder, model_type
        )

        # --- 2. Add Noise and Schedule Timesteps ---
        noisy_model_input, noise, timesteps, sigmas = _add_noise_and_schedule(
            latents, noise_scheduler, noise_random_generator, weighting_scheme,
            logit_mean, logit_std, mode_scale, sp_size, device
        )

        # --- 3. Model Forward Pass ---
        model_pred = _model_forward(
            transformer, model_type, noisy_model_input, prompt_embed, timesteps,
            prompt_attention_mask, audio_embeds, face_mask
        )

        # --- Capture data for potential NaN debug (Moved earlier) ---
        # Note: model_pred and target might be added/updated later if calculated
        data_for_nan_debug = {
            "latents": latents.cpu().detach(), "captions": captions,
            "audio_embeds": audio_embeds.cpu().detach() if audio_embeds is not None else None,
            "face_mask_input": face_mask.cpu().detach(), "sample_ids": sample_ids,
            "cond_latents": cond_latents.cpu().detach(),
            "semantic_images": semantic_images, # Store the whole list of images/None
            "prompt_embed": prompt_embed.cpu().detach(),
            "prompt_attention_mask": prompt_attention_mask.cpu().detach(),
            "noise": noise.cpu().detach(), "timesteps": timesteps.cpu().detach(),
            "sigmas": sigmas.cpu().detach(), "noisy_model_input": noisy_model_input.cpu().detach(),
            # model_pred and target will be added below or during error handling
        }

        # --- Check for NaN in model_pred ---
        is_pred_nan = torch.isnan(model_pred).any()
        # skip_tensor = torch.tensor(1 if is_pred_nan else 0, device=device, dtype=torch.float32)
        # torch.distributed.all_reduce(skip_tensor, op=torch.distributed.ReduceOp.SUM)
        # if skip_tensor.item() > 0:
        #     optimizer.zero_grad()
        #     break

        if is_pred_nan:
            logger.error(f"Rank {rank}: NaN detected in model_pred at step {step}, micro-batch {i+1}/{gradient_accumulation_steps}.")
            logger.error(f"Rank {rank}: Sample IDs in this micro-batch: {sample_ids}")
            logger.error(f"Rank {rank}: Saving debug data to {nan_debug_dir}...")
            local_skip = torch.tensor(is_pred_nan, device=device, dtype=torch.float32)
            torch.distributed.all_reduce(local_skip, op=torch.distributed.ReduceOp.SUM)
            nan_detected_in_step = True
            optimizer.zero_grad()

            # Add model_pred to debug data
            data_for_nan_debug["model_pred"] = model_pred.cpu().detach()
            # Target wasn't calculated, maybe save noise instead or mark as unavailable
            data_for_nan_debug["target"] = None # Target not calculated

            _save_debug_info(
                is_nan=True, # Indicate NaN occurred
                nan_debug_dir=nan_debug_dir,
                sample_loss_dir=sample_loss_dir, # Still save sample info if possible
                rank=rank,
                step=step,
                micro_batch_idx=i,
                data_for_nan_debug=data_for_nan_debug,
                loss_value=torch.tensor(float('nan')), # Loss wasn't calculated
                squared_error=torch.full_like(model_pred, float('nan')), # Placeholder
                background_mask=torch.zeros_like(model_pred[0, 0, 0]), # Placeholder
                face_region_mask=torch.zeros_like(model_pred[0, 0, 0]), # Placeholder
                lip_region_mask=torch.zeros_like(model_pred[0, 0, 0]), # Placeholder
                spatial_loss_map=torch.full_like(model_pred[0, 0, 0], float('nan')), # Placeholder
                weighted_spatial_loss_map=torch.full_like(model_pred[0, 0, 0], float('nan')), # Placeholder
                sample_ids=sample_ids,
                logger=logger
            )
            logger.error(f"Rank {rank}: Skipping loss calculation and backward pass for this micro-batch due to NaN in model_pred.")
            break # Skip to the next micro-batch

        # --- 4. Calculate Target ---
        target = _calculate_target(
            precondition_outputs, model_type, cond_latents, latents, noise
        )

        # --- Update data_for_nan_debug with model_pred and target ---
        # (Do this *after* NaN check on model_pred but before loss calculation)
        data_for_nan_debug["model_pred"] = model_pred.cpu().detach()
        data_for_nan_debug["target"] = target.cpu().detach()

        # --- 5. Calculate and Accumulate Losses ---
        (loss, squared_error, spatial_loss_map, weighted_spatial_loss_map,
         background_mask, face_region_mask, lip_region_mask,
         accum_bg_loss, accum_face_loss, accum_lip_loss,
         accum_bg_count, accum_face_count, accum_lip_count
        ) = _calculate_and_accumulate_losses(
            model_pred, target, face_mask, gradient_accumulation_steps,
            accum_bg_loss, accum_face_loss, accum_lip_loss,
            accum_bg_count, accum_face_count, accum_lip_count, epsilon
        )

        # --- Check for NaN Loss ---
        is_loss_nan = torch.isnan(loss)
        # skip_tensor = torch.tensor(1 if is_loss_nan else 0, device=device, dtype=torch.float32)
        # torch.distributed.all_reduce(skip_tensor, op=torch.distributed.ReduceOp.SUM)
        # if skip_tensor.item() > 0:
        #     optimizer.zero_grad()
        #     break
        
        if is_loss_nan:
            logger.error(f"Rank {rank}: NaN loss detected AFTER calculation at step {step}, micro-batch {i+1}/{gradient_accumulation_steps}.")
            logger.error(f"Rank {rank}: Sample IDs in this micro-batch: {sample_ids}")
            logger.error(f"Rank {rank}: Saving debug data to {nan_debug_dir}...")
            nan_detected_in_step = True
            optimizer.zero_grad()

            # --- 6. Save Debug Info (NaN data, Sample Losses, Heatmaps) ---
            # Call _save_debug_info here as well for NaN loss post-calculation
            _save_debug_info(
                is_nan=True, # Indicate NaN occurred
                nan_debug_dir=nan_debug_dir,
                sample_loss_dir=sample_loss_dir,
                rank=rank,
                step=step,
                micro_batch_idx=i,
                data_for_nan_debug=data_for_nan_debug, # Now includes model_pred and target
                loss_value=loss, # The actual NaN loss value
                squared_error=squared_error,
                background_mask=background_mask,
                face_region_mask=face_region_mask,
                lip_region_mask=lip_region_mask,
                spatial_loss_map=spatial_loss_map,
                weighted_spatial_loss_map=weighted_spatial_loss_map,
                sample_ids=sample_ids,
                logger=logger
            )

            logger.error(f"Rank {rank}: Skipping backward pass and subsequent micro-batches for step {step} due to NaN loss.")
            break # Stop processing micro-batches for this step
        else:
            # --- 6. Save Debug Info (Sample Losses, Heatmaps - Non-NaN case) ---
             _save_debug_info(
                 is_nan=False, # Indicate NOT NaN
                 nan_debug_dir=nan_debug_dir, # Still pass for potential future use? Or None? Pass for consistency.
                 sample_loss_dir=sample_loss_dir,
                 rank=rank,
                 step=step,
                 micro_batch_idx=i,
                 data_for_nan_debug=data_for_nan_debug, # Includes model_pred and target
                 loss_value=loss, # The calculated loss value
                 squared_error=squared_error,
                 background_mask=background_mask,
                 face_region_mask=face_region_mask,
                 lip_region_mask=lip_region_mask,
                 spatial_loss_map=spatial_loss_map,
                 weighted_spatial_loss_map=weighted_spatial_loss_map,
                 sample_ids=sample_ids,
                 logger=logger
             )

        # Note: Removed redundant nan_detected_in_step check here, break statement above handles it

        # --- 7. Backward Pass ---
        # Loss is already scaled by gradient_accumulation_steps
        try:
            loss.backward()
        except Exception as e:
            logger.error(f"Rank {rank}: Error during backward pass at step {step}, micro-batch {i+1}: {e}")
            logger.error(f"Rank {rank}: Sample IDs in this micro-batch: {sample_ids}")
            # Save debug info even on backward error
            _save_debug_info(
                is_nan=True, # Treat backward error like NaN for debugging purposes
                nan_debug_dir=nan_debug_dir,
                sample_loss_dir=sample_loss_dir,
                rank=rank,
                step=step,
                micro_batch_idx=i,
                data_for_nan_debug=data_for_nan_debug, # Includes model_pred and target
                loss_value=loss, # The loss value before backward error
                squared_error=squared_error,
                background_mask=background_mask,
                face_region_mask=face_region_mask,
                lip_region_mask=lip_region_mask,
                spatial_loss_map=spatial_loss_map,
                weighted_spatial_loss_map=weighted_spatial_loss_map,
                sample_ids=sample_ids,
                logger=logger
            )
            logger.error(f"Rank {rank}: Skipping subsequent micro-batches and optimizer step for step {step}.")
            nan_detected_in_step = True # Treat backward error like NaN
            break # Stop processing micro-batches for this step

        # --- Accumulate Total Loss (Based on Original Method) ---
        total_loss += loss.detach().item() # Accumulate the scaled loss
        successful_micro_batches += 1

    # --- End of Gradient Accumulation Loop ---

    grad_norm = None
    avg_bg_loss_batch = 0.0
    avg_face_loss_batch = 0.0
    avg_lip_loss_batch = 0.0

    if not nan_detected_in_step:
        # --- 8. Aggregate and Compute Final Regional Losses ---
        avg_bg_loss_batch, avg_face_loss_batch, avg_lip_loss_batch = _aggregate_and_compute_final_losses(
            accum_bg_loss, accum_face_loss, accum_lip_loss,
            accum_bg_count, accum_face_count, accum_lip_count
        )

        # --- 9. Optimizer Step and LR Update ---
        grad_norm, clipping_error = _optimizer_step_and_lr_update(
            transformer, optimizer, lr_scheduler, max_grad_norm, logger, rank, step
        )
        if clipping_error:
            nan_detected_in_step = True # Treat clipping error like NaN for return purposes

    else:
        # NaN or backward/clipping error detected, skip optimizer step and zero gradients
        optimizer.zero_grad()
        logger.warning(f"Rank {rank}: Optimizer step skipped for step {step} due to error detected in a micro-batch.")
        # Ensure metrics reflect the skipped step
        total_loss = 0.0 # Or should we average over successful_micro_batches? Setting to 0 for clarity.
        grad_norm = 0.0
        # Regional losses remain 0 as they weren't aggregated/computed

    # Return values including the NaN flag and successful micro-batch count
    # Average the total loss over the number of successful micro-batches
    avg_total_loss = total_loss # total_loss is already sum of per-microbatch avg losses
    if successful_micro_batches > 0:
         # Note: total_loss already accumulates the loss scaled by grad_accum_steps
         # So, the average loss for the *entire step* is just total_loss
         # (Sum of (micro_loss / grad_accum) * grad_accum = Sum of micro_loss) / grad_accum = total_loss
         # Let's keep total_loss as the sum of the scaled micro-batch losses.
         # The logging outside will interpret this.
         pass # avg_total_loss = total_loss # Keep as sum of scaled losses

    return avg_total_loss, grad_norm, avg_bg_loss_batch, avg_face_loss_batch, avg_lip_loss_batch, nan_detected_in_step, successful_micro_batches


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=2400)) 
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    from torch.distributed.device_mesh import init_device_mesh
    device_mesh = init_device_mesh("cuda", (world_size,))
    initialize_sequence_parallel_state(args.sp_size)

    # --- 修改检查点恢复逻辑 ---
    # 1. Rank 0 确定要恢复的检查点路径
    determined_resume_path = None
    if rank == 0:
        if args.output_dir is not None and os.path.isdir(args.output_dir):
            checkpoints = [
                d for d in os.listdir(args.output_dir)
                if os.path.isdir(os.path.join(args.output_dir, d)) 
                and d.startswith("checkpoint-")
            ]
            checkpoints.sort(key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else -1)
            
            if checkpoints:
                # 找到检查点
                latest_checkpoint_name = checkpoints[-1]
                if args.resume_from_checkpoint == "auto" or args.resume_from_checkpoint is None:
                    determined_resume_path = os.path.join(args.output_dir, latest_checkpoint_name)
                    main_print(f"检测到检查点，将自动恢复最新检查点: {determined_resume_path}")
                elif os.path.isdir(args.resume_from_checkpoint):
                    # 用户指定了有效目录
                    determined_resume_path = args.resume_from_checkpoint
                    main_print(f"使用用户指定的检查点路径: {determined_resume_path}")
                else:
                     # 用户指定了无效路径，但存在检查点
                     main_print(f"警告：用户指定的恢复路径 {args.resume_from_checkpoint} 无效，但检测到检查点。将不会恢复。")
                     # determined_resume_path 保持为 None
            else:
                # 未找到检查点
                if args.resume_from_checkpoint is not None and args.resume_from_checkpoint != "auto":
                    main_print(f"警告：指定了恢复检查点 {args.resume_from_checkpoint} 但目录中不存在检查点，将从头开始训练")
                    # determined_resume_path 保持为 None
                elif args.resume_from_checkpoint == "auto":
                     main_print(f"警告：设置了自动恢复检查点，但在目录 {args.output_dir} 中未找到检查点，将从头开始训练")
                     # determined_resume_path 保持为 None
                # else: # resume_from_checkpoint is None，正常从头开始
        elif args.resume_from_checkpoint is not None:
            # 输出目录无效，但用户指定了恢复路径
            main_print(f"警告：输出目录 {args.output_dir} 无效或不存在，无法恢复检查点 {args.resume_from_checkpoint}。将从头开始训练。")
            # determined_resume_path 保持为 None

    # 2. 将确定的路径广播到所有 Rank
    resume_path_list = [determined_resume_path]
    dist.broadcast_object_list(resume_path_list, src=0)
    actual_resume_path = resume_path_list[0]
    # ------------------------------------

    # If passed along, set the training seed now. On GPU...
    if args.seed is not None:
        # TODO: t within the same seq parallel group should be the same. Noise should be different.
        set_seed(args.seed + rank)
    # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.
    noise_random_generator = None

    # Handle the repository creation
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weights to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.

    # 加载vae
    main_print(f"--> loading vae from {args.vae_model_name_or_path}")
    vae, autocast_type, fps = load_vae(args.model_type, args.vae_model_name_or_path)
    vae = vae.to(device)
    vae.enable_tiling()

    if args.i2v_mode:
        image_embed_interleave = 4
    else:
        image_embed_interleave = 1

    text_encoder = load_text_encoder(args.model_type, args.pretrained_model_name_or_path, device, args)

    if "audio" in args.model_type:
        if "i2v" in args.model_type:
            train_dataset = LatentDatasetAudio_i2v(args.data_json_path, args.num_latent_t)
            collate_fn = latent_collate_function_audio_i2v
        else:
            train_dataset = LatentDatasetAudio(args.data_json_path, args.num_latent_t,
                                  args.cfg)
            collate_fn = latent_collate_function_audio
    else:
        train_dataset = LatentDataset(args.data_json_path, args.num_latent_t,
                                  args.cfg)
        collate_fn = latent_collate_function
    sampler = (LengthGroupedSampler(
        args.train_batch_size,
        rank=rank,
        world_size=world_size,
        lengths=train_dataset.lengths,
        group_frame=args.group_frame,
        group_resolution=args.group_resolution,
    ) if (args.group_frame or args.group_resolution) else DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=False))

    if "audio_i2v" in args.model_type:
        collate_fn = latent_collate_function_audio_i2v
    elif "audio" in args.model_type:
        collate_fn = latent_collate_function_audio
    else:
        collate_fn = latent_collate_function
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    # Create model:

    main_print(f"--> loading model from {args.pretrained_model_name_or_path}")
    # keep the master weight to float32
    transformer = load_transformer(
        args.model_type,
        args.dit_model_name_or_path,
        args.pretrained_model_name_or_path,
        torch.float32 if args.master_weight_type == "fp32" else torch.bfloat16,
    )

    
    all_params = list(transformer.parameters()) # 获取 FSDP 模型的所有参数
    if args.train_audio_only:
        main_print("--> 只训练与音频相关的参数")
        audio_param_patterns = [
            "audio_scale", "audio_norm", "audio_attn_qkv", "audio_attn_q_norm",
            "audio_attn_k_norm", "audio_proj"
        ]
        audio_params_to_optimize = []
        frozen_params = []
        audio_param_ids = set() # 用于快速查找

        # 先找出所有音频参数
        for name, param in transformer.named_parameters(): # 遍历 FSDP 模型参数
            is_audio_param = False
            for pattern in audio_param_patterns:
                if pattern in name:
                    is_audio_param = True
                    break
            if is_audio_param and param.requires_grad: # 确保参数本身是可训练的
                audio_params_to_optimize.append(param)
                audio_param_ids.add(id(param))
                # main_print(f"  Optimizing: {name}") # 打印确认

        # 将剩余参数视为冻结
        for param in all_params:
            if id(param) not in audio_param_ids:
                frozen_params.append(param)
                # 最好也确保这些参数不需要梯度
                # param.requires_grad_(False) # 可以在 FSDP 包装前做，或者在这里确认

        total_train_params = sum(p.numel() for p in audio_params_to_optimize)
        main_print(f"  音频相关训练参数数量 = {total_train_params / 1e6} M")

        # 使用参数组初始化优化器
        param_groups = [
            {'params': audio_params_to_optimize, 'lr': args.learning_rate},
            {'params': frozen_params, 'lr': 0.0} # 冻结参数的学习率设为 0
        ]
        main_print(f"  Optimizing {len(audio_params_to_optimize)} audio params.")
        main_print(f"  Freezing {len(frozen_params)} other params.")
    else:
        params_to_optimize = [p for p in all_params if p.requires_grad]
        total_train_params = sum(p.numel() for p in params_to_optimize)
        main_print(f"  Total training parameters = {total_train_params / 1e6} M")
        param_groups = [{'params': params_to_optimize, 'lr': args.learning_rate}]

    if args.use_lora:
        assert args.model_type != "hunyuan", "LoRA is only supported for huggingface model. Please use hunyuan_hf for lora finetuning"
        if args.model_type == "mochi":
            pipe = MochiPipeline
        elif args.model_type == "hunyuan_hf":
            pipe = HunyuanVideoPipeline
        transformer.requires_grad_(False)
        transformer_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights=True,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        transformer.add_adapter(transformer_lora_config)

    if args.resume_from_lora_checkpoint:
        lora_state_dict = pipe.lora_state_dict(
            args.resume_from_lora_checkpoint)
        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v
            for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(
            transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer,
                                                      transformer_state_dict,
                                                      adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys",
                                      None)
            if unexpected_keys:
                main_print(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. ")

    main_print(
        f"  Total training parameters = {total_train_params} M"
    )
    main_print(
        f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}"
    )
    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_startegy,
        args.use_lora,
        args.use_cpu_offload,
        args.master_weight_type,
        train_subset_params=args.train_audio_only,
    )

    if args.use_lora:
        transformer.config.lora_rank = args.lora_rank
        transformer.config.lora_alpha = args.lora_alpha
        transformer.config.lora_target_modules = [
            "to_k", "to_q", "to_v", "to_out.0"
        ]
        transformer._no_split_modules = [
            no_split_module.__name__ for no_split_module in no_split_modules
        ]
        fsdp_kwargs["auto_wrap_policy"] = fsdp_kwargs["auto_wrap_policy"](
            transformer)

    # Ensure all parameters have the target dtype before FSDP
    target_dtype = torch.bfloat16 if args.master_weight_type != "fp32" else torch.float32
    main_print(f"--> Casting transformer to {target_dtype} before FSDP initialization...")
    transformer = transformer.to(target_dtype)
    main_print(f"--> Verifying parameter dtypes after cast...")
    all_correct_dtype = True
    for name, param in transformer.named_parameters():
        if param.dtype != target_dtype:
            main_print(f"    WARNING: Parameter {name} has dtype {param.dtype} instead of {target_dtype}")
            all_correct_dtype = False
    if all_correct_dtype:
        main_print(f"--> All parameters successfully cast to {target_dtype}.")


    transformer = FSDP(
        transformer,
        device_mesh=device_mesh,
        **fsdp_kwargs,
    )
    main_print("--> model loaded")

    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(transformer, no_split_modules,
                                 args.selective_checkpointing)

    # Set model as trainable.
    transformer.train()

    noise_scheduler = FlowMatchEulerDiscreteScheduler()


    optimizer = torch.optim.AdamW(
        param_groups, # 使用参数组
        # lr 在组里定义了，这里可以不写，或者写一个默认值（会被组覆盖）
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    init_steps = 0
    # --- 修改检查点恢复逻辑 (Part 2: 使用广播的路径，并增加容错) ---
    if actual_resume_path: # 检查广播来的路径是否有效
        try:
            if args.use_lora:
                # LoRA 恢复逻辑可能需要单独处理或调整 resume_lora_optimizer 函数
                # 假设 resume_lora_optimizer 可以接受具体路径
                # TODO: Add similar try-except for LoRA resume if needed
                transformer, optimizer, init_steps = resume_lora_optimizer(
                    transformer, actual_resume_path, optimizer) # 直接使用路径
                main_print(f"--> 成功从 LoRA 检查点恢复: {actual_resume_path}, 初始step: {init_steps}")
            else:
                transformer, optimizer, init_steps = resume_checkpoint(
                    transformer, optimizer, actual_resume_path) # 直接使用路径
                main_print(f"--> 成功从 FSDP 检查点恢复: {actual_resume_path}, 初始step: {init_steps}")
        except FileNotFoundError as e:
            main_print(f"警告: 尝试从 {actual_resume_path} 恢复检查点失败: {e}")
            main_print("--> 将从头开始训练 (step 0)。")
            init_steps = 0 # 明确设置为 0
        except Exception as e:
            # Catch other potential errors during loading
            main_print(f"警告: 恢复检查点时发生意外错误: {e}")
            main_print("--> 将从头开始训练 (step 0)。")
            init_steps = 0
    else:
        # 如果 actual_resume_path 是 None (由 Rank 0 决定)
        main_print("--> 未找到或未指定有效检查点，将从头开始训练")
        # init_steps 保持为 0
    # ------------------------------------
    main_print(f"optimizer: {optimizer}")

    # --- 在初始化 scheduler 之前，为 optimizer 的 param_groups 添加 initial_lr ---
    # 这是因为从 checkpoint 恢复 optimizer state 后，initial_lr 可能会丢失，
    # 而某些 scheduler (如 LambdaLR) 需要它。
    # 我们根据创建 optimizer 时使用的 lr 来设置 initial_lr。
    if actual_resume_path: # 只在恢复 checkpoint 后执行
        main_print("--> Restoring initial_lr for optimizer param groups before scheduler initialization.")
        initial_lrs = [args.learning_rate, 0.0] # 基于 optimizer 创建时的 param_groups
        if len(optimizer.param_groups) == len(initial_lrs):
            for i, group in enumerate(optimizer.param_groups):
                group['initial_lr'] = initial_lrs[i]
                main_print(f"    Set initial_lr={initial_lrs[i]} for param_group {i}")
        else:
             main_print(f"Warning: Mismatch in number of param groups. Expected {len(initial_lrs)}, found {len(optimizer.param_groups)}. Cannot set initial_lr correctly.")
    # -------------------------------------------------------------------------

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

    # lr_scheduler = get_cosine_schedule_with_warmup(
    # optimizer,
    # num_warmup_steps=args.lr_warmup_steps,
    # num_training_steps=args.max_train_steps,
    # num_cycles=0.5,        
    # last_epoch=init_steps - 1,
    # )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps *
        args.sp_size / args.train_sp_batch_size)
    args.num_train_epochs = math.ceil(args.max_train_steps /
                                      num_update_steps_per_epoch)

    if rank <= 0:
        project = args.tracker_project_name or "fastvideo"
        wandb.init(project=project, config=args)

    # Train!
    total_batch_size = (world_size * args.gradient_accumulation_steps /
                        args.sp_size * args.train_sp_batch_size)
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Num Epochs = {args.num_train_epochs}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(
        f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(
        f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}"
    )
    main_print(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps = {args.max_train_steps}")
    main_print(
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )
    # print dtype
    main_print(
        f"  Master weight dtype: {transformer.parameters().__next__().dtype}"
    )

    # Create sample loss directory on all ranks
    sample_loss_dir = None
    if args.output_dir:
        sample_loss_dir = os.path.join(args.output_dir, "sample_losses")
        # Use rank to prevent race condition in makedirs, though exist_ok should handle it
        # Delaying the actual makedirs until it's needed might be safer in some distributed filesystems
        # For now, let rank 0 ensure it exists, others assume it will.
        # Alternative: all ranks call makedirs with exist_ok=True
        if rank == 0:
             os.makedirs(sample_loss_dir, exist_ok=True)
        dist.barrier() # Ensure rank 0 creates dir before others proceed
        if rank == 0: # Log only once
            main_print(f"--> Sample losses will be saved to: {sample_loss_dir}")

    # Create NaN debug data directory on all ranks
    nan_debug_dir = None
    if args.output_dir:
        nan_debug_dir = os.path.join(args.output_dir, "nan_debug_data")
        if rank == 0:
             os.makedirs(nan_debug_dir, exist_ok=True)
        dist.barrier() # Ensure rank 0 creates dir before others proceed
        if rank == 0: # Log only once
            main_print(f"--> NaN debug data will be saved to: {nan_debug_dir}")


    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        assert NotImplementedError(
            "resume_from_checkpoint is not supported now.")
        # TODO

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=init_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=local_rank > 0,
    )

    if "audio_i2v" in args.model_type:
        loader = sp_parallel_dataloader_wrapper_audio_i2v(
            train_dataloader,
            device,
            args.train_batch_size,
            args.sp_size,
            args.train_sp_batch_size,
        )
    elif "audio" in args.model_type:
        loader = sp_parallel_dataloader_wrapper_audio(
            train_dataloader,
            device,
            args.train_batch_size,
            args.sp_size,
            args.train_sp_batch_size,
        )
    else:
        loader = sp_parallel_dataloader_wrapper(
            train_dataloader,
            device,
            args.train_batch_size,
            args.sp_size,
            args.train_sp_batch_size)

    step_times = deque(maxlen=100)

    # transformer = None
    # optimizer = None
    # lr_scheduler = None
    # todo future
    scaler = GradScaler()
    for i in range(init_steps):
        next(loader)
    for step in range(init_steps + 1, args.max_train_steps + 1):
        start_time = time.time()
        loss, grad_norm, avg_bg_loss, avg_face_loss, avg_lip_loss, nan_detected_in_step, successful_micro_batches = train_one_step(
            transformer,
            args.model_type,
            optimizer,
            lr_scheduler,
            loader,
            noise_scheduler,
            noise_random_generator,
            args.gradient_accumulation_steps,
            args.sp_size,
            args.precondition_outputs,
            args.max_grad_norm,
            args.weighting_scheme,
            args.logit_mean,
            args.logit_std,
            args.mode_scale,
            vae,
            text_encoder,
            args,
            device,
            logger,
            rank,
            args.output_dir,
            step,
            nan_debug_dir,
            scaler,
        )

        # --- Handle NaN detection --- 
        if nan_detected_in_step:
            if rank == 0: # Log only once per step
                logger.warning(f"Optimizer step skipped for step {step} due to NaN detected in a micro-batch.")
            # Note: train_one_step already returns 0 loss/grad_norm in this case
            # We still proceed to logging these values (which will be 0)

        lr = optimizer.param_groups[0]['lr']
        step_time = time.time() - start_time
        step_times.append(step_time)
        avg_step_time = sum(step_times) / len(step_times)

        progress_bar.set_postfix({
            "lr": f"{lr:.8f}",
            "loss": f"{loss:.4f}",
            "step_time": f"{step_time:.2f}s",
            "grad_norm": grad_norm,
        })
        progress_bar.update(1)
        
        
        # 获取最新的检查点路径
        latest_checkpoint = None
        if step > args.checkpointing_steps:
            checkpoint_step = (step // args.checkpointing_steps) * args.checkpointing_steps
            latest_checkpoint = os.path.join(args.output_dir, f"checkpoint-{checkpoint_step}")
            if not os.path.exists(latest_checkpoint):
                latest_checkpoint = None
        
        if rank <= 0:
            wandb.log(
                {
                    "train_loss": loss,
                    "train_loss/background": avg_bg_loss,
                    "train_loss/face": avg_face_loss,
                    "train_loss/lip": avg_lip_loss,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "step_time": step_time,
                    "avg_step_time": avg_step_time,
                    "grad_norm": grad_norm,
                    "step_skipped": int(nan_detected_in_step),
                },
                step=step,
            )
        if step % args.checkpointing_steps == 0:
            if args.use_lora:
                # Save LoRA weights
                save_lora_checkpoint(transformer, optimizer, rank,
                                     args.output_dir, step, pipe)
            else:
                # Your existing checkpoint saving code
                save_checkpoint(transformer, optimizer, rank,
                               args.output_dir, step,
                               train_audio_only=args.train_audio_only)
            dist.barrier()
        if args.log_validation and step % args.validation_steps == 0:
            log_validation(args,
                           transformer,
                           device,
                           torch.bfloat16,
                           step,
                           shift=args.shift)

    if args.use_lora:
        save_lora_checkpoint(transformer, optimizer, rank, args.output_dir,
                             args.max_train_steps, pipe)
    else:
        save_checkpoint(transformer, optimizer, rank, args.output_dir,
                        args.max_train_steps,
                        train_audio_only=args.train_audio_only)


    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()

def add_extra_models_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        title="Extra models args, including vae, text encoders and tokenizers)"
    )

    group.add_argument(
        "--vae-model-name-or-path",
        type=str,
        default=None,
        help="Path to the vae model.",
    )

    group.add_argument(
        "--vae-precision",
        type=str,
        default="fp16",
        choices=PRECISIONS,
        help="Precision mode for the vae model.",
    )

    group.add_argument(
        "--sematic-cond-drop-p",
        type=float,
        default=0.1,
        help="Drop probability for the semantic condition."
    )

    group.add_argument(
        "--i2v-mode",
        action="store_true",
        help="Whether to open i2v mode."
    )
    group.add_argument(
        "--reproduce",
        action="store_true",
        help="Whether to reproduce the training."
    )
    group.add_argument(
        "--text-encoder",
        type=str,
        default="llm-i2v",
        choices=list(TEXT_ENCODER_PATH),
        help="Name of the text encoder model.",
    )
    group.add_argument(
        "--text-encoder-precision",
        type=str,
        default="fp16",
        choices=PRECISIONS,
        help="Precision mode for the text encoder model.",
    )
    group.add_argument(
        "--text-states-dim",
        type=int,
        default=4096,
        help="Dimension of the text encoder hidden states.",
    )
    group.add_argument(
        "--text-len", type=int, default=256, help="Maximum length of the text input."
    )
    group.add_argument(
        "--tokenizer",
        type=str,
        default="llm-i2v",
        choices=list(TOKENIZER_PATH),
        help="Name of the tokenizer model.",
    )
    group.add_argument(
        "--prompt-template",
        type=str,
        default="dit-llm-encode-i2v",
        choices=PROMPT_TEMPLATE,
        help="Image prompt template for the decoder-only text encoder model.",
    )
    group.add_argument(
        "--prompt-template-video",
        type=str,
        default="dit-llm-encode-video-i2v",
        choices=PROMPT_TEMPLATE,
        help="Video prompt template for the decoder-only text encoder model.",
    )
    group.add_argument(
        "--hidden-state-skip-layer",
        type=int,
        default=2,
        help="Skip layer for hidden states.",
    )
    group.add_argument(
        "--apply-final-norm",
        action="store_true",
        help="Apply final normalization to the used text encoder hidden states.",
    )

    # - CLIP
    group.add_argument(
        "--text-encoder-2",
        type=str,
        default="clipL",
        choices=list(TEXT_ENCODER_PATH),
        help="Name of the second text encoder model.",
    )
    group.add_argument(
        "--text-encoder-precision-2",
        type=str,
        default="fp16",
        choices=PRECISIONS,
        help="Precision mode for the second text encoder model.",
    )
    group.add_argument(
        "--text-states-dim-2",
        type=int,
        default=768,
        help="Dimension of the second text encoder hidden states.",
    )
    group.add_argument(
        "--tokenizer-2",
        type=str,
        default="clipL",
        choices=list(TOKENIZER_PATH),
        help="Name of the second tokenizer model.",
    )
    group.add_argument(
        "--text-len-2",
        type=int,
        default=77,
        help="Maximum length of the second text input.",
    )

    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default="mochi",
        help=
        "The type of model to train. Currentlt support [mochi, hunyuan_hf, hunyuan, hunyuan_audio]"
    )
    # dataset & dataloaderyou
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument("--num_height", type=int, default=480)
    parser.add_argument("--num_width", type=int, default=848)
    parser.add_argument("--num_frames", type=int, default=163)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help=
        "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_latent_t",
                        type=int,
                        default=28,
                        help="Number of latent timesteps.")
    parser.add_argument("--group_frame", action="store_true")  # TODO
    parser.add_argument("--group_resolution", action="store_true")  # TODO

    # text encoder & vae & diffusion model
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dit_model_name_or_path", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=0.1)
    parser.add_argument(
        "--precondition_outputs",
        action="store_true",
        help="Whether to precondition the outputs of the model.",
    )

    # validation & logs
    parser.add_argument("--validation_prompt_dir", type=str)
    parser.add_argument("--uncond_prompt_dir", type=str)
    parser.add_argument(
        "--validation_sampling_steps",
        type=str,
        default="64",
        help="use ',' to split multi sampling steps",
    )
    parser.add_argument(
        "--validation_guidance_scale",
        type=str,
        default="4.5",
        help="use ',' to split multi scale",
    )
    parser.add_argument("--validation_steps", type=int, default=50)
    parser.add_argument("--log_validation", action="store_true")
    parser.add_argument("--tracker_project_name", type=str, default=None)
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=
        "The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=
        ("Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
         " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
         " training using `--resume_from_checkpoint`."),
    )
    parser.add_argument("--shift",
                        type=float,
                        default=1.0,
                        help=("Set shift to 7 for hunyuan model."))
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=
        ("Whether training should be resumed from a previous checkpoint. Use a path saved by"
         ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
         ),
    )
    parser.add_argument(
        "--resume_from_lora_checkpoint",
        type=str,
        default=None,
        help=
        ("Whether training should be resumed from a previous lora checkpoint. Use a path saved by"
         ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
         ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=
        ("[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
         " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."),
    )

    # optimizer & scheduler & Training
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help=
        "Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help=
        "Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=15,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=
        "Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=
        ("Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
         " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
         ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=
        ("Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
         " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
         " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
         ),
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help=
        "Whether to use CPU offload for param & gradient & optimizer states.",
    )

    parser.add_argument("--sp_size",
                        type=int,
                        default=1,
                        help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )

    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=False,
        help="Whether to use LoRA for finetuning.",
    )
    parser.add_argument("--lora_alpha",
                        type=int,
                        default=256,
                        help="Alpha parameter for LoRA.")
    parser.add_argument("--lora_rank",
                        type=int,
                        default=128,
                        help="LoRA rank parameter. ")
    parser.add_argument("--fsdp_sharding_startegy", default="full")

    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="uniform",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "uniform"],
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="mean to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="std to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help=
        "Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=
        ('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
         ' "constant", "constant_with_warmup"]'),
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.01,
                        help="Weight decay to apply.")
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )
    parser.add_argument(
        "--train_audio_only",
        action="store_true",
        help="只训练与音频相关的参数",
    )

    parser = add_extra_models_args(parser)

    args = parser.parse_args()
    main(args)
