# import
import os
import json
import torch
import torch.distributed as dist
from torch.distributed import checkpoint as dist_cp
from peft import get_peft_model_state_dict
from safetensors.torch import load_file, save_file
import stat
from torch.distributed.checkpoint.default_planner import (DefaultLoadPlanner,
                                                          DefaultSavePlanner)
from torch.distributed.checkpoint.optimizer import \
    load_sharded_optimizer_state_dict
from torch.distributed.fsdp import (FullOptimStateDictConfig,
                                    FullStateDictConfig)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

from fastvideo.utils.logging_ import main_print
from torch.distributed.checkpoint import save as distcp_save, load as distcp_load
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict, set_model_state_dict,
    get_optimizer_state_dict, set_optimizer_state_dict,
)
from pathlib import Path
import logging # Using logging is generally better than print
import sys
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP # Assuming FSDP type
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

# Basic logging setup (can be configured further)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def save_checkpoint_optimizer(model,
                              optimizer,
                              rank,
                              output_dir,
                              step,
                              discriminator=False):
    with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    ):
        cpu_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(
            model,
            optimizer,
        )

    # todo move to get_state_dict
    save_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(save_dir, exist_ok=True)
    # save using safetensors
    if rank <= 0 and not discriminator:
        weight_path = os.path.join(save_dir,
                                   "diffusion_pytorch_model.safetensors")
        save_file(cpu_state, weight_path)
        config_dict = dict(model.config)
        config_dict.pop('dtype')
        config_path = os.path.join(save_dir, "config.json")
        # save dict as json
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        optimizer_path = os.path.join(save_dir, "optimizer.pt")
        torch.save(optim_state, optimizer_path)
    else:
        weight_path = os.path.join(save_dir,
                                   "discriminator_pytorch_model.safetensors")
        save_file(cpu_state, weight_path)
        optimizer_path = os.path.join(save_dir, "discriminator_optimizer.pt")
        torch.save(optim_state, optimizer_path)
    main_print(f"--> checkpoint saved at step {step}")


# def save_checkpoint(
#     transformer: FSDP,
#     optimizer: torch.optim.Optimizer,
#     rank: int,
#     output_dir: str,
#     step: int,
#     train_audio_only: bool = False # Keep flag for config
# ):
#     """
#     Saves a sharded FSDP model and optimizer checkpoint using torch.distributed.checkpoint.save API.

#     Args:
#         transformer: The FSDP-wrapped model instance.
#         optimizer: The optimizer instance managing the FSDP model's parameters.
#         rank: The current process rank.
#         output_dir: The base directory where checkpoints will be saved.
#         step: The current training step, used for the checkpoint directory name.
#         train_audio_only: Flag indicating the training mode (saved in config).
#                           The actual saved state will be the full sharded state.
#     """
#     main_print(f"--> Saving sharded checkpoint using dist_cp.save at step {step}")

#     # Define the main directory for this specific checkpoint
#     save_dir = os.path.join(output_dir, f"checkpoint-{step}")
#     # dist_cp.save handles directory creation if the storage writer supports it (like FileSystemWriter)
#     # Ensure the top-level output directory exists (safer on rank 0)
#     if rank == 0:
#         os.makedirs(output_dir, exist_ok=True)
#     dist.barrier() # Ensure directory exists before proceeding

#     # --- Get Model and Optimizer State Dicts ---
#     # No FSDP context needed here, use the new API directly
#     # model_state = {} # No longer needed
#     # optim_state = {} # No longer needed

#     # Get state dicts for model and optimizer using specific functions
#     # Note: These functions work correctly with FSDP models
#     model_state = get_model_state_dict(transformer)
#     optim_state = get_optimizer_state_dict(model=transformer, optimizers=optimizer)

#     # Combine into a single state dict for saving
#     # Using descriptive keys for clarity
#     state_dict_to_save = {
#         "model_state": model_state,
#         "optimizer_state": optim_state,
#     }
#     main_print(f"  --> Obtained model and optimizer state dicts.")


#     # --- Save Combined State Dict ---
#     dist_cp.save(
#         state_dict=state_dict_to_save,
#         checkpoint_id=save_dir # Use the directory path as checkpoint_id for FileSystemWriter
#         # storage_writer defaults to FileSystemWriter when checkpoint_id is a path
#         # planner defaults to DefaultSavePlanner
#         # no_dist=False is default
#     )
#     main_print(f"  --> Combined state dict saved to {save_dir}")


#     # --- Save Configuration (only on Rank 0) ---
#     # This part remains largely the same, saving metadata.
#     if rank == 0:
#         config_dict = {}
#         # Access the original model config potentially wrapped by FSDP
#         base_model = transformer.module if isinstance(transformer, FSDP) else transformer
#         if hasattr(base_model, "config") and base_model.config is not None:
#             try:
#                 config_dict = dict(base_model.config)
#                 # Remove non-serializable items like dtype if present
#                 if "dtype" in config_dict:
#                     del config_dict["dtype"]
#                 # Add any other custom config items if needed
#             except Exception as e:
#                  main_print(f"Warning: Could not serialize model config: {e}")
#                  config_dict = {} # Fallback to empty dict
#         else:
#              main_print("Warning: Base model does not have a 'config' attribute.")

#         # Crucially, save the training mode flag for loading purposes
#         config_dict["train_audio_only"] = train_audio_only

#         # 添加优化器参数组元数据
#         optimizer_params = [
#             {
#                 "lr": group["lr"],
#                 "betas": group["betas"],
#                 "eps": group["eps"],
#                 "weight_decay": group["weight_decay"],
#                 "initial_lr": group.get("initial_lr", group["lr"])  # 捕获初始学习率
#             }
#             for group in optimizer.param_groups
#         ]

#         # Prepare the final dictionary to save as JSON
#         config_to_save = {
#             "step": step,
#             "model_config": config_dict,
#             "optimizer_params": optimizer_params  # 新增优化器参数配置
#         }

#         config_save_path = os.path.join(save_dir, "config.json")
#         try:
#             with open(config_save_path, "w") as f:
#                 json.dump(config_to_save, f, indent=4)
#             main_print(f"  --> Configuration saved to {config_save_path}")
#         except Exception as e:
#             main_print(f"Error saving config.json to {config_save_path}: {e}")

#     # Barrier to ensure all ranks have completed their part of the save
#     # before declaring the checkpoint fully saved.
#     dist.barrier()

#     main_print(f"--> Sharded checkpoint saved successfully using dist_cp.save at step {step} to {save_dir}")


# def save_checkpoint(
#     transformer: FSDP,
#     optimizer: torch.optim.Optimizer,
#     rank: int,
#     output_dir: str,
#     step: int,
#     train_audio_only: bool = False,      # 保持兼容但仅写入 config
# ):
#     """
#     使用 torch.distributed.checkpoint.save 保存 FSDP + Optimizer 分片权重。
#     目录结构与旧版保持一致：output_dir/checkpoint-{step}/
#     额外写入 config.json 记录元数据。
#     """
#     save_dir = Path(output_dir) / f"checkpoint-{step}"
#     # ---------- 0. rank-0 创建目录 ----------
#     if rank == 0:
#         save_dir.mkdir(parents=True, exist_ok=True)
#     dist.barrier()

#     # ---------- 1. 收集统一 state_dict ----------
#     #   - DTensor 是默认格式；可按需通过 settings 调整
#     state_dict = get_state_dict(
#         model=transformer,
#         optim={"optimizer": optimizer},
#     )
#     # ---------- 2. 实际写盘 ----------
#     # checkpoint_id 直接给目录路径；FileSystemWriter 自动分片
#     distcp_save(state_dict, checkpoint_id=str(save_dir))
#     main_print(f"--> [Rank {rank}] State-dict saved to {save_dir}")

#     # ---------- 3. rank-0 额外保存配置 ----------
#     if rank == 0:
#         config = {
#             "step": step,
#             "train_audio_only": train_audio_only,
#             "optimizer_param_groups": [
#                 {
#                     "lr": g["lr"],
#                     "betas": g["betas"],
#                     "eps": g["eps"],
#                     "weight_decay": g["weight_decay"],
#                     "initial_lr": g.get("initial_lr", g["lr"]),
#                 }
#                 for g in optimizer.param_groups
#             ],
#         }
#         with (save_dir / "config.json").open("w") as f:
#             json.dump(config, f, indent=4)
#         main_print(f"--> [Rank 0] Metadata written to {save_dir/'config.json'}")

#     dist.barrier()
#     main_print(f"--> Checkpoint-{step} completed")

def save_checkpoint(
    transformer: FSDP,
    optimizer: torch.optim.Optimizer,
    rank: int,
    output_dir: str,
    step: int,
    train_audio_only: bool = False,
):
    """
    Saves a sharded FSDP checkpoint including model and optimizer states.

    Args:
        transformer: The FSDP-wrapped model.
        optimizer: The optimizer instance.
        rank: The current distributed rank.
        output_dir: The base directory where checkpoints will be saved.
        step: The current training step, used for naming the checkpoint subdirectory.
        train_audio_only: Metadata flag to save.
    
    Raises:
        OSError: If directory creation fails on rank 0.
        Exception: If distcp_save or metadata writing fails.
    """
    save_dir = Path(output_dir) / f"checkpoint-{step}"
    
    # Rank 0 creates the directory, handle potential errors
    if rank == 0:
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            save_dir.chmod(stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            logger.info(f"[Rank {rank}] Created checkpoint directory: {save_dir}")
        except OSError as e:
            logger.error(f"[Rank {rank}] Failed to create checkpoint directory {save_dir}: {e}")
            # Signal other ranks about the failure before they hang on the barrier
            # A simple way is to let the exception propagate after the barrier,
            # but broadcasting an error object might be cleaner in complex scenarios.
            dist.barrier() # Wait for others to potentially reach here
            raise e # Re-raise the exception

    dist.barrier() # Ensure directory exists before proceeding

    # --- Get State Dicts ---
    # Use sharded state dicts with CPU offloading to prevent OOM
    # opts = StateDictOptions(full_state_dict=True,  cpu_offload=True) # NEW - Keep on GPU initially
    try:
        logger.info(f"[Rank {rank}] Getting model state_dict")
        module_state_dict, optimizer_state_dict = get_state_dict(transformer, optimizer)
    except Exception as e:
         logger.error(f"[Rank {rank}] Failed to get state dicts: {e}")
         dist.barrier() # Sync before potential exit
         raise e

    # --- Save State Dicts (Distributed) ---
    state_dict_to_save = {"model": module_state_dict, "optim": optimizer_state_dict}
    try:
        logger.info(f"[Rank {rank}] Starting distributed checkpoint save to {save_dir}...")
        distcp_save(state_dict_to_save, checkpoint_id=str(save_dir))
        logger.info(f"[Rank {rank}] Finished distributed checkpoint save to {save_dir}.")
    except Exception as e:
        logger.error(f"[Rank {rank}] Failed during distcp_save to {save_dir}: {e}")
        # distcp_save is collective, an error likely affects all ranks
        dist.barrier() # Sync before potential exit
        raise RuntimeError(f"distcp_save failed on rank {rank}") from e

    # --- Save Metadata (Rank 0 Only) ---
    if rank == 0:
        optimizer_groups = [
                # Ensure only JSON-serializable types are saved
                {k: (list(v) if isinstance(v, tuple) else v) 
                 for k, v in g.items() if k in ("lr", "betas", "eps", "weight_decay")}
                for g in optimizer.param_groups
            ]
        
        # 添加初始学习率
        for g in optimizer.param_groups:
            g["initial_lr"] = g.get("initial_lr", g["lr"])

        meta = {
            "step": step,
            "train_audio_only": train_audio_only,
            "optimizer_class": optimizer.__class__.__name__, # Save optimizer class name
            "optimizer_groups": optimizer_groups,
            # Add other relevant metadata like FSDP config if needed
            "fsdp_config": str(transformer.auto_wrap_policy) if hasattr(transformer, 'auto_wrap_policy') else "N/A", 
        }
        meta_path = save_dir / "config.json"
        try:
            meta_path.write_text(json.dumps(meta, indent=4))
            logger.info(f"[Rank {rank}] Saved metadata to {meta_path}")
        except (OSError, IOError) as e:
            logger.error(f"[Rank {rank}] Failed to write metadata file {meta_path}: {e}")
            # This error only happens on rank 0, but it means the checkpoint is incomplete.
            # We should probably signal this failure. For simplicity, we raise here.
            # Consider cleanup (deleting save_dir) if needed.
            dist.barrier() # Allow other ranks to finish saving before rank 0 raises
            raise e
        except TypeError as e:
             logger.error(f"[Rank {rank}] Failed to serialize metadata to JSON: {e}")
             dist.barrier()
             raise e


    dist.barrier() # Final barrier to ensure saving is complete everywhere
    logger.info(f"[Rank {rank}] Checkpoint saving process complete for step {step}.")



def save_checkpoint_generator_discriminator(
    model,
    optimizer,
    discriminator,
    discriminator_optimizer,
    rank,
    output_dir,
    step,
):
    with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    ):
        cpu_state = model.state_dict()

    # todo move to get_state_dict
    save_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(save_dir, exist_ok=True)
    hf_weight_dir = os.path.join(save_dir, "hf_weights")
    os.makedirs(hf_weight_dir, exist_ok=True)
    # save using safetensors
    if rank <= 0:
        config_dict = dict(model.config)
        config_path = os.path.join(hf_weight_dir, "config.json")
        # save dict as json
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        weight_path = os.path.join(hf_weight_dir,
                                   "diffusion_pytorch_model.safetensors")
        save_file(cpu_state, weight_path)

    main_print(f"--> saved HF weight checkpoint at path {hf_weight_dir}")
    model_weight_dir = os.path.join(save_dir, "model_weights_state")
    os.makedirs(model_weight_dir, exist_ok=True)
    model_optimizer_dir = os.path.join(save_dir, "model_optimizer_state")
    os.makedirs(model_optimizer_dir, exist_ok=True)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        optim_state = FSDP.optim_state_dict(model, optimizer)
        model_state = model.state_dict()
        weight_state_dict = {"model": model_state}
        dist_cp.save_state_dict(
            state_dict=weight_state_dict,
            storage_writer=dist_cp.FileSystemWriter(model_weight_dir),
            planner=DefaultSavePlanner(),
        )
        optimizer_state_dict = {"optimizer": optim_state}
        dist_cp.save_state_dict(
            state_dict=optimizer_state_dict,
            storage_writer=dist_cp.FileSystemWriter(model_optimizer_dir),
            planner=DefaultSavePlanner(),
        )

    discriminator_fsdp_state_dir = os.path.join(save_dir,
                                                "discriminator_fsdp_state")
    os.makedirs(discriminator_fsdp_state_dir, exist_ok=True)
    with FSDP.state_dict_type(
            discriminator,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    ):
        optim_state = FSDP.optim_state_dict(discriminator,
                                            discriminator_optimizer)
        model_state = discriminator.state_dict()
        state_dict = {"optimizer": optim_state, "model": model_state}
        if rank <= 0:
            discriminator_fsdp_state_fil = os.path.join(
                discriminator_fsdp_state_dir, "discriminator_state.pt")
            torch.save(state_dict, discriminator_fsdp_state_fil)

    main_print("--> saved FSDP state checkpoint")


def load_sharded_model(model, optimizer, model_dir, optimizer_dir):
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        weight_state_dict = {"model": model.state_dict()}

        optim_state = load_sharded_optimizer_state_dict(
            model_state_dict=weight_state_dict["model"],
            optimizer_key="optimizer",
            storage_reader=dist_cp.FileSystemReader(optimizer_dir),
        )
        optim_state = optim_state["optimizer"]
        flattened_osd = FSDP.optim_state_dict_to_load(
            model=model, optim=optimizer, optim_state_dict=optim_state)
        optimizer.load_state_dict(flattened_osd)
        dist_cp.load_state_dict(
            state_dict=weight_state_dict,
            storage_reader=dist_cp.FileSystemReader(model_dir),
            planner=DefaultLoadPlanner(),
        )
        model_state = weight_state_dict["model"]
        model.load_state_dict(model_state)
    main_print(f"--> loaded model and optimizer from path {model_dir}")
    return model, optimizer


def load_full_state_model(model, optimizer, checkpoint_file, rank):
    with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    ):
        discriminator_state = torch.load(checkpoint_file)
        model_state = discriminator_state["model"]
        if rank <= 0:
            optim_state = discriminator_state["optimizer"]
        else:
            optim_state = None
        model.load_state_dict(model_state)
        discriminator_optim_state = FSDP.optim_state_dict_to_load(
            model=model, optim=optimizer, optim_state_dict=optim_state)
        optimizer.load_state_dict(discriminator_optim_state)
    main_print(
        f"--> loaded discriminator and discriminator optimizer from path {checkpoint_file}"
    )
    return model, optimizer


def resume_training_generator_discriminator(model, optimizer, discriminator,
                                            discriminator_optimizer,
                                            checkpoint_dir, rank):
    step = int(checkpoint_dir.split("-")[-1])
    model_weight_dir = os.path.join(checkpoint_dir, "model_weights_state")
    model_optimizer_dir = os.path.join(checkpoint_dir, "model_optimizer_state")
    model, optimizer = load_sharded_model(model, optimizer, model_weight_dir,
                                          model_optimizer_dir)
    discriminator_ckpt_file = os.path.join(checkpoint_dir,
                                           "discriminator_fsdp_state",
                                           "discriminator_state.pt")
    discriminator, discriminator_optimizer = load_full_state_model(
        discriminator, discriminator_optimizer, discriminator_ckpt_file, rank)
    return model, optimizer, discriminator, discriminator_optimizer, step


def resume_training(model, optimizer, checkpoint_dir, discriminator=False):
    weight_path = os.path.join(checkpoint_dir,
                               "diffusion_pytorch_model.safetensors")
    if discriminator:
        weight_path = os.path.join(checkpoint_dir,
                                   "discriminator_pytorch_model.safetensors")
    model_weights = load_file(weight_path)

    with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    ):
        current_state = model.state_dict()
        current_state.update(model_weights)
        model.load_state_dict(current_state, strict=False)
    if discriminator:
        optim_path = os.path.join(checkpoint_dir, "discriminator_optimizer.pt")
    else:
        optim_path = os.path.join(checkpoint_dir, "optimizer.pt")
    optimizer_state_dict = torch.load(optim_path, weights_only=False)
    optim_state = FSDP.optim_state_dict_to_load(
        model=model, optim=optimizer, optim_state_dict=optimizer_state_dict)
    optimizer.load_state_dict(optim_state)
    step = int(checkpoint_dir.split("-")[-1])
    return model, optimizer, step


def save_lora_checkpoint(transformer, optimizer, rank, output_dir, step,
                         pipeline):
    with FSDP.state_dict_type(
            transformer,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        full_state_dict = transformer.state_dict()
        lora_optim_state = FSDP.optim_state_dict(
            transformer,
            optimizer,
        )

    if rank <= 0:
        save_dir = os.path.join(output_dir, f"lora-checkpoint-{step}")
        os.makedirs(save_dir, exist_ok=True)

        # save optimizer
        optim_path = os.path.join(save_dir, "lora_optimizer.pt")
        torch.save(lora_optim_state, optim_path)
        # save lora weight
        main_print(f"--> saving LoRA checkpoint at step {step}")
        transformer_lora_layers = get_peft_model_state_dict(
            model=transformer, state_dict=full_state_dict)
        pipeline.save_lora_weights(
            save_directory=save_dir,
            transformer_lora_layers=transformer_lora_layers,
            is_main_process=True,
        )
        # save config
        lora_config = {
            "step": step,
            "lora_params": {
                "lora_rank": transformer.config.lora_rank,
                "lora_alpha": transformer.config.lora_alpha,
                "target_modules": transformer.config.lora_target_modules,
            },
        }
        config_path = os.path.join(save_dir, "lora_config.json")
        with open(config_path, "w") as f:
            json.dump(lora_config, f, indent=4)
    main_print(f"--> LoRA checkpoint saved at step {step}")


def resume_lora_optimizer(transformer, checkpoint_dir, optimizer):
    # 基础校验并自动查找最新检查点
    if os.path.isfile(checkpoint_dir):
        # 如果是文件路径，直接使用
        checkpoint_dir = checkpoint_dir
    elif os.path.isdir(checkpoint_dir):
        # 如果是目录，查找最新的检查点子目录
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
        if not checkpoints:
            raise FileNotFoundError(f"在目录 {checkpoint_dir} 中未找到有效的检查点")
        
        # 按照检查点编号排序
        checkpoints.sort(key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0)
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
        main_print(f"--> 自动选择最新检查点: {latest_checkpoint}")
        checkpoint_dir = latest_checkpoint
    if not os.path.isdir(checkpoint_dir):
        raise NotADirectoryError(f"检查点路径 {checkpoint_dir} 不是有效目录")

    # 自动检测训练模式
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    optim_path = os.path.join(checkpoint_dir, "lora_optimizer.pt")
    optimizer_state_dict = torch.load(optim_path, weights_only=False)
    optim_state = FSDP.optim_state_dict_to_load(
        model=transformer,
        optim=optimizer,
        optim_state_dict=optimizer_state_dict)
    optimizer.load_state_dict(optim_state)
    step = config_dict["step"]
    main_print(f"-->  Successfully resuming LoRA optimizer from step {step}")
    return transformer, optimizer, step


# def resume_checkpoint(
#     transformer: FSDP,
#     optimizer: torch.optim.Optimizer,
#     checkpoint_dir: str
# ):
#     """
#     Resumes training from a sharded FSDP checkpoint saved using torch.distributed.checkpoint.save API.

#     Args:
#         transformer: The FSDP-wrapped model instance. Must be already wrapped with FSDP.
#         optimizer: The optimizer instance. It MUST be initialized correctly for the
#                    *current* training run (e.g., with correct parameter groups based
#                    on current args.train_audio_only) BEFORE calling this function.
#                    Loading only restores the internal states (momentum, etc.).
#         checkpoint_dir: The path to the specific checkpoint directory
#                         (e.g., /path/to/output/checkpoint-1000) created by the
#                         `save_checkpoint` function using `dist_cp.save`. Auto-detection
#                         of latest checkpoint within a parent directory is supported.

#     Returns:
#         Tuple[FSDP, torch.optim.Optimizer, int]: The model and optimizer with loaded
#                                                  states, and the step number saved in the checkpoint.
#     """
#     rank = dist.get_rank()
#     # --- 1. 查找并验证检查点目录 ---
#     resolved_checkpoint_dir = None
#     if os.path.isdir(checkpoint_dir):
#         # 检查是否是包含多个 checkpoint-* 的父目录
#         potential_checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, d))]
#         if potential_checkpoints:
#             # 如果是父目录，找到最新的 checkpoint 子目录
#             potential_checkpoints.sort(key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else -1, reverse=True)
#             resolved_checkpoint_dir = os.path.join(checkpoint_dir, potential_checkpoints[0])
#             if rank == 0:
#                 print(f"--> Found multiple checkpoints in {checkpoint_dir}. Resuming from latest: {resolved_checkpoint_dir}")
#         else:
#             # 假设传入的就是具体的 checkpoint 目录
#             resolved_checkpoint_dir = checkpoint_dir
#     elif os.path.isfile(checkpoint_dir):
#          # 不支持直接传入文件路径，必须是目录
#          raise ValueError(f"checkpoint_dir must be a directory, not a file: {checkpoint_dir}")
#     else:
#         raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

#     if not os.path.isdir(resolved_checkpoint_dir):
#         raise NotADirectoryError(f"Resolved checkpoint path is not a valid directory: {resolved_checkpoint_dir}")

#     checkpoint_dir = resolved_checkpoint_dir # Use the resolved path

#     config_path = os.path.join(checkpoint_dir, "config.json")
#     # The actual state dict files are managed internally by dist_cp.load within checkpoint_dir

#     if not os.path.isfile(config_path):
#         raise FileNotFoundError(f"Config file not found in checkpoint directory: {config_path}")

#     # --- 2. Load Configuration (Rank 0 reads, then broadcast or all read) ---
#     config_data = {}
#     if rank == 0:
#         with open(config_path, "r") as f:
#             config_data = json.load(f)
#         main_print(f"--> Loaded config from {config_path}")

#     config_list = [config_data]
#     dist.broadcast_object_list(config_list, src=0)
#     if rank != 0:
#         config_data = config_list[0]

#     saved_step = config_data.get("step", 0)
#     saved_train_audio_only = config_data.get("model_config", {}).get("train_audio_only", False) # Informational

#     if rank == 0:
#         print(f"--> Resuming from checkpoint saved at step {saved_step}.")
#         print(f"    Checkpoint was saved with train_audio_only={saved_train_audio_only}.")
#         print(f"    Ensure current optimizer setup matches intended training mode.")

#     # --- 3. 恢复参数组配置（在所有rank上）---
#     optimizer_params = config_data.get("optimizer_params", [])
#     param_group_configs = [optimizer_params] if rank == 0 else [None]
#     dist.broadcast_object_list(param_group_configs, src=0)
#     optimizer_params = param_group_configs[0]

#     for i, group in enumerate(optimizer.param_groups):
#         if i < len(optimizer_params):
#             saved_params = optimizer_params[i]
#             group.update({
#                 k: saved_params[k]
#                 for k in ["initial_lr", "betas", "eps", "weight_decay"]
#                 if k in saved_params
#             })
#     main_print(f"  --> Optimizer parameter groups configured based on checkpoint.")

#     # --- 4. 加载分片状态使用 dist_cp.load ---
#     # Prepare an empty state dict structure to be filled by dist_cp.load
#     # The keys MUST match those used during saving ("model_state", "optimizer_state")
#     state_dict_to_load = {
#         "model_state": {},
#         "optimizer_state": {},
#     }

#     # No FSDP context manager needed here
#     dist_cp.load(
#         state_dict=state_dict_to_load,
#         checkpoint_id=checkpoint_dir # Provide the directory path
#         # storage_reader defaults to FileSystemReader
#         # planner defaults to DefaultLoadPlanner
#         # no_dist=False is default
#     )
#     main_print(f"  --> Loaded combined state dict from {checkpoint_dir}")

#     # --- 5. Apply loaded states using set_state_dict ---
#     set_state_dict(
#         model=transformer,
#         state_dict=state_dict_to_load["model_state"],
#     )
#     main_print(f"  --> Applied loaded state to the model.")

#     set_state_dict(
#         model=optimizer, # Yes, pass the optimizer instance here
#         state_dict=state_dict_to_load["optimizer_state"],
#     )
#     main_print(f"  --> Applied loaded state to the optimizer.")


#     # Barrier not strictly necessary after load but good practice
#     dist.barrier()

#     main_print(f"--> Successfully resumed from sharded checkpoint using dist_cp.load: {checkpoint_dir} at step {saved_step}")

#     return transformer, optimizer, saved_step


def resume_checkpoint(
    transformer: FSDP,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str,
):
    """
    Resumes training from a sharded FSDP checkpoint.

    If checkpoint_dir points to a directory containing multiple 'checkpoint-*' 
    subdirectories, it automatically tries to load the latest one based on the step number.

    Args:
        transformer: The FSDP-wrapped model (must have the same structure as the saved one).
        optimizer: The optimizer instance (must be compatible with the saved state).
        checkpoint_dir: Path to the specific checkpoint directory OR a parent directory 
                        containing multiple 'checkpoint-*' subdirectories.

    Returns:
        tuple: (transformer, optimizer, loaded_step) where loaded_step is the step 
               number loaded from the checkpoint metadata, or 0 if no checkpoint was loaded.
               The transformer and optimizer objects are modified in-place.

    Raises:
        FileNotFoundError: If a valid checkpoint directory cannot be found or is invalid.
        RuntimeError: If loading fails due to distcp errors, metadata issues, or inconsistencies.
        ValueError: If optimizer structure mismatch prevents hyperparameter restoration.
    """
    rank = dist.get_rank()
    ckpt_path = Path(checkpoint_dir)
    load_path: Path | None = None # The final path to load from

    # --- Determine Checkpoint Path ---
    if ckpt_path.is_dir() and not (ckpt_path / "config.json").exists():
        # Parent directory provided: Find the latest checkpoint subdirectory
        latest_ckpt_path = None
        latest_step = -1
        if rank == 0: logger.info(f"[Rank {rank}] Checkpoint directory {ckpt_path} lacks config.json, searching for latest 'checkpoint-*' subdirectory...")
        
        # Search logic only needs to run on rank 0, then broadcast the result
        found_path_str = None
        if rank == 0:
            try:
                potential_ckpts = list(ckpt_path.glob("checkpoint-*"))
                if not potential_ckpts:
                    logger.warning(f"[Rank {rank}] No 'checkpoint-*' subdirectories found in {ckpt_path}.")
                else:
                    for p in potential_ckpts:
                        if p.is_dir():
                            try:
                                step = int(p.name.split("-")[-1])
                                if step > latest_step:
                                    # Check if it looks like a valid checkpoint dir (has .metadata file)
                                    # Note: distcp creates a .metadata file, checking for it is more robust than config.json
                                    if (p / ".metadata").is_file():
                                         latest_step = step
                                         latest_ckpt_path = p
                                    else:
                                         logger.warning(f"[Rank {rank}] Directory {p} looks like a checkpoint but missing '.metadata'. Skipping.")
                            except (ValueError, IndexError):
                                logger.warning(f"[Rank {rank}] Could not parse step number from directory name: {p.name}. Skipping.")
                            except OSError as e:
                                logger.warning(f"[Rank {rank}] Filesystem error checking directory {p}: {e}. Skipping.")
                    
                    if latest_ckpt_path:
                        logger.info(f"[Rank {rank}] Found latest valid checkpoint: {latest_ckpt_path} at step {latest_step}")
                        found_path_str = str(latest_ckpt_path)
                    else:
                         logger.warning(f"[Rank {rank}] No valid 'checkpoint-*' subdirectories with '.metadata' found in {ckpt_path}.")

            except OSError as e:
                logger.error(f"[Rank {rank}] Filesystem error when searching for checkpoints in {ckpt_path}: {e}")
                # Broadcast error signal
                found_path_str = f"ERROR:OSError:{e}"

        # Broadcast the found path (or error) from rank 0 to all ranks
        broadcast_list = [found_path_str]
        dist.broadcast_object_list(broadcast_list, src=0)
        received_path_str = broadcast_list[0]

        if received_path_str is None:
             # No valid checkpoint found by rank 0
             raise FileNotFoundError(f"No valid checkpoint found in {ckpt_path}")
        elif isinstance(received_path_str, str) and received_path_str.startswith("ERROR:"):
             # Rank 0 encountered an error
             error_msg = received_path_str.split(":", 2)[-1]
             logger.error(f"[Rank {rank}] Received error from rank 0 while searching checkpoints: {error_msg}")
             raise RuntimeError(f"Error finding checkpoint on rank 0: {error_msg}")
        else:
            load_path = Path(received_path_str)

    else:
        # Specific checkpoint directory provided (or non-directory path)
        load_path = ckpt_path

    # --- Validate Final Path ---
    if rank == 0: # Validation check primarily on rank 0, but error broadcast ensures consistency
        is_valid = False
        error_msg = ""
        if not load_path or not load_path.is_dir():
            error_msg = f"Checkpoint path {load_path} is not a valid directory."
            logger.error(f"[Rank {rank}] {error_msg}")
        elif not (load_path / "config.json").exists():
             error_msg = f"Checkpoint directory {load_path} is missing config.json."
             logger.error(f"[Rank {rank}] {error_msg}")
        elif not (load_path / ".metadata").exists(): # Also check for distcp metadata
             error_msg = f"Checkpoint directory {load_path} is missing distcp '.metadata' file."
             logger.error(f"[Rank {rank}] {error_msg}")
        else:
             is_valid = True
        
        # Broadcast validation result (or error message)
        validation_result = ["VALID" if is_valid else f"ERROR:{error_msg}"]
        dist.broadcast_object_list(validation_result, src=0)

    else: # Non-zero ranks receive validation result
        validation_result = [None]
        dist.broadcast_object_list(validation_result, src=0)

    final_validation = validation_result[0]
    if isinstance(final_validation, str) and final_validation.startswith("ERROR:"):
        error_msg = final_validation.split(":", 1)[-1]
        if rank != 0: # Log error on non-zero ranks as well
             logger.error(f"[Rank {rank}] Received validation error from rank 0: {error_msg}")
        raise FileNotFoundError(error_msg) # Raise consistent error everywhere
    elif final_validation != "VALID":
         # Should not happen if logic is correct, but as a safeguard
         raise RuntimeError("Checkpoint validation failed unexpectedly.")

    if rank == 0: logger.info(f"[Rank {rank}] Determined checkpoint path to load: {load_path}")

    # --- Load Metadata (Rank 0) and Broadcast ---
    meta = {}
    load_error = None
    if rank == 0:
        try:
            meta_path = load_path / "config.json"
            meta_content = meta_path.read_text()
            meta = json.loads(meta_content)
            logger.info(f"[Rank {rank}] Loaded metadata from {meta_path}")
        except FileNotFoundError:
            load_error = f"config.json not found in {load_path}"
            logger.error(f"[Rank {rank}] {load_error}")
        except json.JSONDecodeError as e:
            load_error = f"Failed to parse config.json in {load_path}: {e}"
            logger.error(f"[Rank {rank}] {load_error}")
        except OSError as e:
            load_error = f"Error reading config.json in {load_path}: {e}"
            logger.error(f"[Rank {rank}] {load_error}")
        except Exception as e: # Catch any other unexpected error
            load_error = f"Unexpected error loading metadata: {e}"
            logger.error(f"[Rank {rank}] {load_error}")

    # Broadcast metadata or error signal
    broadcast_data = [meta if load_error is None else {"error": load_error}]
    dist.broadcast_object_list(broadcast_data, src=0)
    
    received_meta = broadcast_data[0]
    if isinstance(received_meta, dict) and "error" in received_meta:
        error_msg = received_meta["error"]
        if rank != 0:
            logger.error(f"[Rank {rank}] Received error signal from Rank 0 regarding metadata: {error_msg}")
        raise RuntimeError(f"Failed to load checkpoint metadata: {error_msg}")
    
    meta = received_meta # Use the successfully broadcasted metadata
    saved_step = meta.get("step", 0)
    if rank == 0: logger.info(f"[Rank {rank}] Checkpoint metadata indicates step: {saved_step}")

    # --- Prepare State Dict Shells ---
    opts = StateDictOptions(full_state_dict=False, cpu_offload=False) # NEW
    try:
        logger.info(f"[Rank {rank}] Getting model state_dict")
        module_state_dict, optimizer_state_dict = get_state_dict(transformer, optimizer)
    except Exception as e:
         logger.error(f"[Rank {rank}] Failed to get state dict shells for loading: {e}")
         dist.barrier() # Sync before potential exit
         raise e

    # --- Load State Dicts (Distributed) ---
    state_dict_to_load = {"model": module_state_dict, "optim": optimizer_state_dict}
    try:
        logger.info(f"[Rank {rank}] Starting distributed checkpoint load from {load_path}...")
        # distcp_load will load data into the GPU state dict shells
        distcp_load(state_dict_to_load, checkpoint_id=str(load_path))
        logger.info(f"[Rank {rank}] Finished distributed checkpoint load from {load_path}.")
    except Exception as e:
        # distcp_load is collective, errors likely affect all ranks
        logger.error(f"[Rank {rank}] Failed during distcp_load from {load_path}: {e}")
        dist.barrier() # Sync before potential exit
        raise RuntimeError(f"distcp_load failed on rank {rank} from {load_path}") from e

    # --- Set Loaded State Dicts ---
    try:
        logger.info(f"[Rank {rank}] Setting model state_dict")
        # Setting GPU state dict shells back to the model
        # set_model_state_dict(transformer, module_state_dict, options=opts)
        # logger.info(f"[Rank {rank}] Setting optimizer state_dict (on GPU)...")
        # # Pass model for FSDP optimizer state dict setting
        # set_optimizer_state_dict(transformer, optimizer, optimizer_state_dict, options=opts)
        # logger.info(f"[Rank {rank}] State dicts set.")
        set_state_dict(model=transformer, model_state_dict=module_state_dict, optimizers=optimizer, optim_state_dict=optimizer_state_dict)
    except Exception as e:
        logger.error(f"[Rank {rank}] Failed to set loaded state dicts: {e}")
        dist.barrier() # Sync before potential exit
        raise RuntimeError(f"Failed to set state dicts on rank {rank}") from e

    # --- Restore Optimizer Hyperparameters from Metadata ---
    # This ensures LR, weight decay etc. are exactly as saved, overriding any defaults
    # used when creating the optimizer object before calling resume_checkpoint.
    loaded_groups = meta.get("optimizer_groups")
    if loaded_groups:
        if len(loaded_groups) == len(optimizer.param_groups):
            for i, group in enumerate(optimizer.param_groups):
                # Update known hyperparameters found in the loaded metadata group
                for key in ("lr", "betas", "eps", "weight_decay"):
                    if key in loaded_groups[i]:
                        # Convert betas back to tuple if needed (saved as list)
                        value = loaded_groups[i][key]
                        if key == "betas" and isinstance(value, list):
                            value = tuple(value)
                        group[key] = value
            if rank == 0: logger.info(f"[Rank {rank}] Restored optimizer hyperparameters (lr, betas, etc.) from config.json.")
        else:
            # Log error/warning on all ranks as this is a potential issue
            logger.warning(
                f"[Rank {rank}] Optimizer param group count mismatch! "
                f"Loaded metadata has {len(loaded_groups)} groups, "
                f"current optimizer has {len(optimizer.param_groups)} groups. "
                f"Cannot restore hyperparameters from metadata."
            )
            # Depending on severity, you might want to raise an error here:
            # raise ValueError("Optimizer structure mismatch prevents hyperparameter restoration.")
    elif rank == 0:
        logger.warning(f"[Rank {rank}] 'optimizer_groups' not found in config.json. Cannot restore hyperparameters like LR from metadata.")


    dist.barrier() # Ensure all ranks have finished loading and setting states
    if rank == 0:
        logger.info(f"[Rank {rank}] Successfully resumed checkpoint from {load_path} at step {saved_step}")

    # Return the modified objects and the loaded step
    return transformer, optimizer, saved_step