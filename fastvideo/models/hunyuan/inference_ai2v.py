import os
import random
import time
from pathlib import Path
import json # Added for loading config.json if needed later, though not used currently for state loading

import torch
import torch.distributed as dist
# Removed dist_cp imports for save/load_state_dict as we use the newer API
# from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner # Not needed for distcp_load
from torch.distributed.checkpoint import load as distcp_load # Import the new load function
from torch.distributed.checkpoint.state_dict import set_state_dict, get_state_dict # Import the new state_dict functions
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# from torch.distributed.fsdp import StateDictType # Not directly needed for distcp load/set
from loguru import logger
from safetensors.torch import load_file as safetensors_load_file

from fastvideo.models.hunyuan.constants import (NEGATIVE_PROMPT,
                                                NEGATIVE_PROMPT_I2V,
                                                PRECISION_TO_TYPE,
                                                PROMPT_TEMPLATE)
from fastvideo.models.hunyuan.diffusion.pipelines.pipeline_hunyuan_video import HunyuanVideoAIPipeline
from fastvideo.models.hunyuan.diffusion.schedulers import \
    FlowMatchDiscreteScheduler
from fastvideo.models.hunyuan.modules import load_model
from fastvideo.models.hunyuan.modules.posemb_layers import get_nd_rotary_pos_embed
from fastvideo.models.hunyuan.text_encoder import TextEncoder_i2v
from fastvideo.models.hunyuan.utils.data_utils import align_to, get_closest_ratio, generate_crop_size_list
from fastvideo.models.hunyuan.vae import load_vae
from fastvideo.models.hunyuan.utils.lora_utils import load_lora_for_pipeline
from fastvideo.utils.parallel_states import nccl_info
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, Union
import functools

try:
    import xfuser
    from xfuser.core.distributed import (
        get_sequence_parallel_world_size,
        get_sequence_parallel_rank,
        get_sp_group,
        initialize_model_parallel,
        init_distributed_environment
    )
except:
    xfuser = None
    get_sequence_parallel_world_size = None
    get_sequence_parallel_rank = None
    get_sp_group = None
    initialize_model_parallel = None
    init_distributed_environment = None


def get_1d_rotary_pos_embed_riflex(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    k: Optional[int] = None,
    L_test: Optional[int] = None,
):
    """
    RIFLEx: Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        k (`int`, *optional*, defaults to None): the index for the intrinsic frequency in RoPE
        L_test (`int`, *optional*, defaults to None): the number of frames for inference
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    freqs = 1.0 / (
            theta ** (torch.arange(0, dim, 2, device=pos.device)[: (dim // 2)].float() / dim)
    )  # [D/2]

    # === Riflex modification start ===
    # Reduce the intrinsic frequency to stay within a single period after extrapolation (see Eq. (8)).
    # Empirical observations show that a few videos may exhibit repetition in the tail frames.
    # To be conservative, we multiply by 0.9 to keep the extrapolated length below 90% of a single period.
    if k is not None:
        freqs[k-1] = 0.9 * 2 * torch.pi / L_test
    # === Riflex modification end ===

    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis


###############################################

def parallelize_transformer(pipe):
    transformer = pipe.transformer
    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # Should be in range(0, 1000).
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,  # Now we don't use it.
        text_states_2: Optional[torch.Tensor] = None,  # Text embedding for modulation.
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None,  # Guidance for modulation, should be cfg_scale x 1000.
        return_dict: bool = True,
    ):
        if x.shape[-2] // 2 % get_sequence_parallel_world_size() == 0:
            # try to split x by height
            split_dim = -2
        elif x.shape[-1] // 2 % get_sequence_parallel_world_size() == 0:
            # try to split x by width
            split_dim = -1
        else:
            raise ValueError(f"Cannot split video sequence into ulysses_degree x ring_degree ({get_sequence_parallel_world_size()}) parts evenly")

        # patch sizes for the temporal, height, and width dimensions are 1, 2, and 2.
        temporal_size, h, w = x.shape[2], x.shape[3] // 2, x.shape[4] // 2

        x = torch.chunk(x, get_sequence_parallel_world_size(),dim=split_dim)[get_sequence_parallel_rank()]

        dim_thw = freqs_cos.shape[-1]
        freqs_cos = freqs_cos.reshape(temporal_size, h, w, dim_thw)
        freqs_cos = torch.chunk(freqs_cos, get_sequence_parallel_world_size(),dim=split_dim - 1)[get_sequence_parallel_rank()]
        freqs_cos = freqs_cos.reshape(-1, dim_thw)
        dim_thw = freqs_sin.shape[-1]
        freqs_sin = freqs_sin.reshape(temporal_size, h, w, dim_thw)
        freqs_sin = torch.chunk(freqs_sin, get_sequence_parallel_world_size(),dim=split_dim - 1)[get_sequence_parallel_rank()]
        freqs_sin = freqs_sin.reshape(-1, dim_thw)
        
        from xfuser.core.long_ctx_attention import xFuserLongContextAttention
        
        for block in transformer.double_blocks + transformer.single_blocks:
            block.hybrid_seq_parallel_attn = xFuserLongContextAttention()

        output = original_forward(
            x,
            t,
            text_states,
            text_mask,
            text_states_2,
            freqs_cos,
            freqs_sin,
            guidance,
            return_dict,
        )

        return_dict = not isinstance(output, tuple)
        sample = output["x"]
        sample = get_sp_group().all_gather(sample, dim=split_dim)
        output["x"] = sample
        return output

    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward

class Inference(object):

    def __init__(
        self,
        args,
        vae,
        vae_kwargs,
        text_encoder,
        model,
        text_encoder_2=None,
        pipeline=None,
        use_cpu_offload=False,
        device=None,
        logger=None,
        parallel_args=None,
    ):
        self.vae = vae
        self.vae_kwargs = vae_kwargs

        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2

        self.model = model
        self.pipeline = pipeline
        self.use_cpu_offload = use_cpu_offload

        self.args = args
        self.device = (device if device is not None else
                       "cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.parallel_args = parallel_args

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_path,
                        args,
                        device=None,
                        **kwargs):
        """
        Initialize the Inference pipeline.

        Args:
            pretrained_model_path (str or pathlib.Path): The model path, including t2v, text encoder and vae checkpoints.
            args (argparse.Namespace): The arguments for the pipeline.
            device (str or torch.device): The device for inference. Default is "cuda" or "cpu". Changed default behavior description.
        """
        # ========================================================================
        logger.info(
            f"Got text-to-video model root path: {pretrained_model_path}")

        # ==================== Initialize Distributed Environment ================
        # Ensure distributed is initialized if needed for loading sharded checkpoints
        # Handled before calling from_pretrained or determined by nccl_info
        # if args.sp_size > 1 and not dist.is_initialized():
        #      logger.warning("Distributed environment might be required for sharded checkpoints but is not initialized.")

        if nccl_info.sp_size > 1 and dist.is_initialized():
            # Ensure device matches the local rank in distributed setting
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            device = torch.device(f"cuda:{local_rank}")
            logger.info(f"Distributed environment detected (SP size > 1). Setting device to local rank: {device}")
        elif device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
        else:
            # If device is provided (e.g., "cuda:0"), use it directly
            device = torch.device(device)
            logger.info(f"Using provided device: {device}")


        parallel_args = None  # {"ulysses_degree": args.ulysses_degree, "ring_degree": args.ring_degree}

        # ======================== Get the args path =============================

        # Disable gradient
        torch.set_grad_enabled(False)

        # =========================== Build main model ===========================
        logger.info("Building model...")
        factor_kwargs = {
            "device": device, # Pass determined device
            "dtype": PRECISION_TO_TYPE[args.precision]
        }

        # Simplified channel logic based on comments in original code
        in_channels = args.latent_channels
        out_channels = args.latent_channels
        image_embed_interleave = 1 # Default, adjusted later if i2v_mode requires it

        if args.i2v_mode:
            if args.i2v_condition_type == "latent_concat":
                in_channels = args.latent_channels * 2 + 1
                image_embed_interleave = 2
            elif args.i2v_condition_type == "token_replace":
                # in_channels remains latent_channels
                image_embed_interleave = 4
            # Update text encoder args specifically for i2v
            args.text_encoder = "llm-i2v"
            args.tokenizer = "llm-i2v"
            args.prompt_template = "dit-llm-encode-i2v"
            args.prompt_template_video = "dit-llm-encode-video-i2v"


        model = load_model(
            args,
            in_channels=in_channels,
            out_channels=out_channels,
            factor_kwargs=factor_kwargs,
        )

        # Move model to device *before* potential FSDP wrapping or loading state dict
        logger.info(f"Moved initial model structure to device: {device}")

        # --- Potential FSDP Wrapping ---
        # If using FSDP for inference, wrap the model *here*, before loading state dict.
        # Example (assuming args.use_fsdp_inference exists and is True):
        # if args.use_fsdp_inference:
        #     from torch.distributed.fsdp.wrap import auto_wrap, transformer_auto_wrap_policy
        #     # Define your FSDP policy, e.g., based on transformer blocks
        #     # my_auto_wrap_policy = functools.partial(
        #     #     transformer_auto_wrap_policy, transformer_layer_cls={ YourTransformerBlockClass }
        #     # )
        #     fsdp_config = {
        #         "auto_wrap_policy": my_auto_wrap_policy, # Replace with your policy
        #         "device_id": device, # Specify the device ID
        #         "use_orig_params": True # Often recommended for newer PyTorch versions
        #         # Add other FSDP parameters as needed (sharding strategy, etc.)
        #     }
        #     logger.info("Wrapping model with FSDP for inference...")
        #     model = FSDP(model, **fsdp_config)
        #     if dist.is_initialized():
        #         dist.barrier()
        #     logger.info("Model wrapped with FSDP.")
        # else:
        #     logger.info("FSDP wrapping for inference is not enabled.")
        # Make sure 'model' is the potentially FSDP-wrapped model before passing to load_state_dict

        # --- Load State Dict ---
        # Pass the (potentially FSDP-wrapped) model and determined device
        model = Inference.load_state_dict(args, model, pretrained_model_path, device)
        model.eval()
        model = model.to(device)

        # ============================= Build extra models ========================
        # VAE
        vae_device = device if not args.use_cpu_offload else torch.device("cpu")
        logger.info(f"Loading VAE to device: {vae_device}")
        vae, _, s_ratio, t_ratio = load_vae(
            args.vae,
            args.vae_precision,
            logger=logger,
            device=vae_device,
        )
        vae_kwargs = {"s_ratio": s_ratio, "t_ratio": t_ratio}

        # Text encoder (adjust prompts and settings based on args)
        # i2v settings were moved up before model creation

        # prompt_template
        prompt_template = (PROMPT_TEMPLATE[args.prompt_template]
                           if args.prompt_template is not None else None)

        # prompt_template_video
        prompt_template_video = (PROMPT_TEMPLATE[args.prompt_template_video]
                                 if args.prompt_template_video is not None else
                                 None)

        crop_start = 0
        if prompt_template_video is not None:
            crop_start = prompt_template_video.get("crop_start", 0)
        elif prompt_template is not None:
            crop_start = prompt_template.get("crop_start", 0)

        max_length = args.text_len + crop_start

        text_encoder_device = device if not args.use_cpu_offload else torch.device("cpu")
        logger.info(f"Loading TextEncoder_i2v to device: {text_encoder_device}")
        text_encoder = TextEncoder_i2v(
            text_encoder_type=args.text_encoder,
            max_length=max_length,
            text_encoder_precision=args.text_encoder_precision,
            tokenizer_type=args.tokenizer,
            i2v_mode=args.i2v_mode,
            prompt_template=prompt_template,
            prompt_template_video=prompt_template_video,
            hidden_state_skip_layer=args.hidden_state_skip_layer,
            apply_final_norm=args.apply_final_norm,
            reproduce=args.reproduce,
            logger=logger,
            device=text_encoder_device,
            image_embed_interleave=image_embed_interleave # Pass determined value
        )

        text_encoder_2 = None
        if args.text_encoder_2 is not None:
            text_encoder_2_device = device if not args.use_cpu_offload else torch.device("cpu")
            logger.info(f"Loading TextEncoder_i2v (secondary) to device: {text_encoder_2_device}")
            text_encoder_2 = TextEncoder_i2v(
                text_encoder_type=args.text_encoder_2,
                max_length=args.text_len_2,
                text_encoder_precision=args.text_encoder_precision_2,
                tokenizer_type=args.tokenizer_2,
                reproduce=args.reproduce,
                logger=logger,
                device=text_encoder_2_device,
                # Assuming secondary encoder doesn't need i2v specific args unless specified
            )

        return cls(
            args=args,
            vae=vae,
            vae_kwargs=vae_kwargs,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            model=model, # Pass the loaded model
            use_cpu_offload=args.use_cpu_offload,
            device=device, # Pass the main determined device
            logger=logger,
            parallel_args=parallel_args,
        )

    @staticmethod
    def load_state_dict(args, model, pretrained_model_path, device):
        """
        Loads the model state dict. Supports loading:
        1. Sharded checkpoints saved via `torch.distributed.checkpoint.save` (preferred).
        2. Single weight files (.pt, .safetensors).

        Args:
            args: Command line arguments. Expected to have `dit_weight`.
            model: The model instance (potentially FSDP wrapped).
            pretrained_model_path: Base path if `dit_weight` is not provided (legacy fallback).
            device: The target device (used for single file loading).

        Returns:
            The model with loaded state dict.

        Raises:
            FileNotFoundError: If a valid weight path cannot be determined or found.
            RuntimeError: If loading the sharded checkpoint fails.
            ValueError: If an unsupported file format is encountered.
        """
        load_key = args.load_key # Used for nested dicts in single-file legacy checkpoints
        dit_weight_path_arg = args.dit_weight

        load_path: Path | None = None
        is_new_sharded_format = False
        is_single_file = False

        # --- Determine Load Path and Format ---
        if dit_weight_path_arg:
            potential_path = Path(dit_weight_path_arg)
            if potential_path.is_dir():
                # Check if it's a new format sharded checkpoint directory
                # New format requires .metadata and config.json (as saved by our save_checkpoint)
                metadata_file = potential_path / ".metadata"
                config_file = potential_path / "config.json" # Optional check, .metadata is key
                if metadata_file.is_file():
                    logger.info(f"Detected new sharded checkpoint directory (found .metadata): {potential_path}")
                    load_path = potential_path
                    is_new_sharded_format = True
                else:
                    # It's a directory, but not the new sharded format. Look for single files inside (legacy behavior).
                    logger.warning(f"Directory specified in --dit-weight ({potential_path}) is not a new sharded checkpoint (missing .metadata). Looking for single weight files inside...")
                    files = list(potential_path.glob("*.safetensors")) + list(potential_path.glob("*.pt"))
                    if files:
                        # Try standard names first
                        found = False
                        for fname in [f"pytorch_model_{load_key}.pt", "diffusion_pytorch_model.safetensors", "model.safetensors"]:
                             fpath = potential_path / fname
                             if fpath.is_file():
                                 load_path = fpath
                                 is_single_file = True
                                 found = True
                                 logger.info(f"Found single weight file inside directory: {load_path}")
                                 break
                        if not found:
                             load_path = files[0] # Fallback to first found file
                             is_single_file = True
                             logger.warning(f"Could not find standard weight names, loading first found file: {load_path}")
                    else:
                         logger.warning(f"No model weights (.pt or .safetensors) found in directory: {potential_path}")
                         # Continue searching in legacy path below if load_path is still None

            elif potential_path.is_file():
                # It's a direct path to a single weight file
                if potential_path.suffix in [".pt", ".safetensors"]:
                    load_path = potential_path
                    is_single_file = True
                    logger.info(f"Using single weight file specified by --dit-weight: {load_path}")
                else:
                     logger.warning(f"Specified --dit-weight path is a file but not .pt or .safetensors: {potential_path}")
            else:
                logger.warning(f"Specified --dit-weight path does not exist: {potential_path}")

        # --- Fallback to Legacy Path Structure (if --dit-weight didn't yield a valid path) ---
        if load_path is None:
            logger.warning("--dit-weight did not provide a valid path or was not specified. Attempting legacy path structure based on `pretrained_model_path`.")
            # Example legacy structure: pretrained_model_path / f"t2v_{args.model_resolution}"
            # You might need to adjust this based on your actual legacy structure
            model_dir = Path(pretrained_model_path) / f"t2v_{args.model_resolution}"
            if model_dir.is_dir():
                files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.pt"))
                if files:
                    found = False
                    for fname in [f"pytorch_model_{load_key}.pt", "diffusion_pytorch_model.safetensors", "model.safetensors"]:
                         fpath = model_dir / fname
                         if fpath.is_file():
                             load_path = fpath
                             is_single_file = True
                             found = True
                             logger.info(f"Found legacy weight file: {load_path}")
                             break
                    if not found:
                        load_path = files[0] # Fallback
                        is_single_file = True
                        logger.warning(f"Could not find standard legacy weight names in {model_dir}, loading first found file: {load_path}")
                else:
                     logger.warning(f"No model weights found in legacy path: {model_dir}")
            else:
                 logger.warning(f"Legacy model directory does not exist: {model_dir}")


        if load_path is None:
            raise FileNotFoundError("Could not determine a valid model weight path to load from --dit-weight or legacy paths.")

        # --- Loading Logic ---
        if is_new_sharded_format:
            logger.info(f"Loading new format sharded model state from: {load_path}")
            if not dist.is_initialized():
                 # This should ideally be checked earlier, but add a safeguard
                 logger.error("Distributed environment is not initialized. Cannot load sharded checkpoint.")
                 raise RuntimeError("Distributed environment must be initialized to load sharded checkpoints.")

            if not isinstance(model, FSDP):
                 logger.warning("Loading a sharded checkpoint, but the model is not wrapped with FSDP. This might lead to errors or unexpected behavior if the checkpoint was saved from an FSDP model.")
                 # Attempt loading anyway, but it might fail.

            # Prepare state dict shell using get_state_dict (safer for FSDP)
            # Inference only needs model state, optimizer state is None
            try:
                logger.info("Getting model state dict structure...")
                # We only need the model's state dict structure for loading.
                # get_state_dict expects optimizer=None if not loading optimizer.
                module_state_dict, _ = get_state_dict(model, [])
                state_dict_to_load = {"model": module_state_dict} # Match the key used in save_checkpoint
                logger.info("Model state dict structure obtained.")
            except Exception as e:
                logger.error(f"Failed to get model state dict structure: {e}")
                raise RuntimeError("Could not get model state dict structure for loading.") from e


            # Load the state dict distributively
            try:
                distcp_load(
                    state_dict=state_dict_to_load, # Load directly into the prepared structure
                    checkpoint_id=str(load_path)   # Directory path
                )
                logger.info(f"distcp_load successfully loaded data from {load_path}")
            except Exception as e:
                 logger.error(f"Failed to load sharded state dict using distcp_load from {load_path}: {e}")
                 raise RuntimeError(f"Could not load sharded checkpoint from {load_path}") from e

            # Apply the loaded state dict to the model
            try:
                # We only need to set the model state dict. Optimizer is None.
                set_state_dict(
                    model=model,
                    model_state_dict=state_dict_to_load["model"], # Use the loaded model state
                    optimizers=[], # No optimizer state to set
                    optim_state_dict={} # No optimizer state to set
                )
                logger.info(f"Successfully applied loaded sharded model state using set_state_dict.")
            except Exception as e:
                 logger.error(f"Failed to set loaded sharded model state dict: {e}")
                 raise RuntimeError("Could not set loaded sharded model state dict") from e

            # Barrier to ensure all ranks finish loading before proceeding
            if dist.is_initialized():
                dist.barrier()
                logger.info("Dist barrier after loading sharded checkpoint.")

        elif is_single_file:
            # Load Single File Checkpoint (.pt or .safetensors)
            logger.info(f"Loading single-file model state from: {load_path} to device: {device}...")
            state_dict = None
            try:
                if load_path.suffix == ".safetensors":
                    # Load directly to target device if possible
                    state_dict = safetensors_load_file(str(load_path), device=str(device))
                elif load_path.suffix == ".pt":
                    # map_location handles device placement
                    state_dict = torch.load(str(load_path), map_location=device)
                else:
                    # This case should not be reached due to earlier checks, but as a safeguard:
                    raise ValueError(f"Unsupported file format (should be .pt or .safetensors): {load_path}")
            except Exception as e:
                 logger.error(f"Failed to load single-file checkpoint from {load_path}: {e}")
                 raise RuntimeError(f"Could not load weights from {load_path}") from e


            # Handle potential nesting (e.g., {'module': ..., 'ema': ...}, or custom key)
            processed_state_dict = state_dict
            if isinstance(state_dict, dict):
                 keys = list(state_dict.keys())
                 # Prioritize load_key if provided and present
                 if load_key and load_key in keys:
                     logger.info(f"Extracting state dict with key '{load_key}'")
                     processed_state_dict = state_dict[load_key]
                 # Common keys from DDP/FSDP saves if load_key not found or not applicable
                 elif "module" in keys:
                      logger.warning(f"Key '{load_key}' not found or not provided, but 'module' key exists. Using 'module' state dict.")
                      processed_state_dict = state_dict["module"]
                 elif "model" in keys and len(keys) == 1: # Check if it's just {"model": ...}
                     logger.info("Found 'model' key as the only key, using its value.")
                     processed_state_dict = state_dict["model"]
                 elif "ema" in keys: # Example: handle 'ema' key if needed
                     logger.warning(f"Using 'ema' state dict found in the checkpoint.")
                     processed_state_dict = state_dict["ema"]
                 # Add more potential keys if necessary

            # Check if the final state dict is actually a dict
            if not isinstance(processed_state_dict, dict):
                 logger.error(f"Loaded state dict (after potential unwrapping) is not a dictionary. Type: {type(processed_state_dict)}")
                 raise TypeError("Processed state_dict is not a dictionary, cannot load.")


            # Load the state dict into the model
            try:
                 if isinstance(model, FSDP) and state_dict is processed_state_dict:
                     # If model is FSDP and we are loading a raw (non-nested) state dict
                     logger.warning("Model is FSDP wrapped, loading a potentially non-flattened single-file checkpoint. Ensure state_dict keys match FSDP parameters.")
                     # FSDP's load_state_dict might handle this if keys match flattened params.
                     # Consider using set_state_dict with full_state_dict=True if issues arise.

                 missing_keys, unexpected_keys = model.load_state_dict(processed_state_dict, strict=False)
                 if missing_keys:
                     logger.warning(f"Missing keys during state_dict load: {missing_keys}")
                 if unexpected_keys:
                     logger.warning(f"Unexpected keys during state_dict load: {unexpected_keys}")
                 logger.info(f"Successfully loaded single-file model state into the model from {load_path}")
            except Exception as e:
                 logger.error(f"Failed to load state_dict into model: {e}")
                 # Log some keys for debugging
                 if isinstance(processed_state_dict, dict):
                     logger.error(f"State dict keys (first 10): {list(processed_state_dict.keys())[:10]}")
                 if hasattr(model, "state_dict"):
                    logger.error(f"Model keys (first 10): {list(model.state_dict().keys())[:10]}")
                 raise RuntimeError(f"Error applying loaded state dict from {load_path} to model.") from e

        else:
             # This should not happen if logic above is correct
             raise RuntimeError("Reached unexpected state: Could not determine loading method (sharded or single file).")


        return model

    @staticmethod
    def parse_size(size):
        if isinstance(size, int):
            size = [size]
        if not isinstance(size, (list, tuple)):
            raise ValueError(
                f"Size must be an integer or (height, width), got {size}.")
        if len(size) == 1:
            size = [size[0], size[0]]
        if len(size) != 2:
            raise ValueError(
                f"Size must be an integer or (height, width), got {size}.")
        return size


class HunyuanAI2VideoSampler(Inference):
    """
    支持音频输入的视频采样器
    
    继承自HunyuanVideoSampler，增加了对audio_embeds的处理支持
    """
    
    def __init__(
        self,
        args,
        vae,
        vae_kwargs,
        text_encoder,
        model,
        text_encoder_2=None,
        pipeline=None,
        use_cpu_offload=False,
        device=0,
        logger=None,
        parallel_args=None,
    ):
        super().__init__(
            args,
            vae,
            vae_kwargs,
            text_encoder,
            model,
            text_encoder_2=text_encoder_2,
            pipeline=pipeline,
            use_cpu_offload=use_cpu_offload,
            device=device,
            logger=logger,
            parallel_args=parallel_args,
        )

        self.pipeline = self.load_diffusion_pipeline(
            args=args,
            vae=self.vae,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            model=self.model,
            device=self.device,
        )

        if args.i2v_mode:
            self.default_negative_prompt = NEGATIVE_PROMPT_I2V
            if args.use_lora:
                self.pipeline = load_lora_for_pipeline(
                    self.pipeline, args.lora_path, LORA_PREFIX_TRANSFORMER="Hunyuan_video_I2V_lora", alpha=args.lora_scale,
                    device=self.device,
                    is_parallel=(self.parallel_args['ulysses_degree'] > 1 or self.parallel_args['ring_degree'] > 1))
                logger.info(f"load lora {args.lora_path} into pipeline, lora scale is {args.lora_scale}.")
        else:
            self.default_negative_prompt = NEGATIVE_PROMPT

    def load_diffusion_pipeline(
        self,
        args,
        vae,
        text_encoder,
        text_encoder_2,
        model,
        scheduler=None,
        device=None,
        progress_bar_config=None,
        data_type="video",
    ):
        """加载音频条件视频生成的推理pipeline"""
        if scheduler is None:
            if args.denoise_type == "flow":
                scheduler = FlowMatchDiscreteScheduler(
                    shift=args.flow_shift,
                    reverse=args.flow_reverse,
                    solver=args.flow_solver,
                )
            else:
                raise ValueError(f"Invalid denoise type {args.denoise_type}")

        # 注意这里使用HunyuanVideoAIPipeline替代了HunyuanVideoPipeline
        pipeline = HunyuanVideoAIPipeline(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            transformer=model,
            scheduler=scheduler,
            progress_bar_config=progress_bar_config,
            args=args,
        )
        if self.use_cpu_offload:
            pipeline.enable_sequential_cpu_offload()
        else:
            pipeline = pipeline.to(device)

        return pipeline

    def get_rotary_pos_embed(self, video_length, height, width):
        target_ndim = 3
        ndim = 5 - 2  # B, C, F, H, W -> F, H, W
    
        # Compute latent sizes based on VAE type
        if "884" in self.args.vae:
            latents_size = [(video_length - 1) // 4 + 1, height // 8, width // 8]
        elif "888" in self.args.vae:
            latents_size = [(video_length - 1) // 8 + 1, height // 8, width // 8]
        else:
            latents_size = [video_length, height // 8, width // 8]
    
        # Compute rope sizes
        if isinstance(self.model.patch_size, int):
            assert all(s % self.model.patch_size == 0 for s in latents_size), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.model.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // self.model.patch_size for s in latents_size]
        elif isinstance(self.model.patch_size, list):
            assert all(
                s % self.model.patch_size[idx] == 0
                for idx, s in enumerate(latents_size)
            ), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.model.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // self.model.patch_size[idx] for idx, s in enumerate(latents_size)]
    
        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # Pad time axis
    
        # 20250316 pftq: Add RIFLEx logic for > 192 frames
        L_test = rope_sizes[0]  # Latent frames
        L_train = 25  # Training length from HunyuanVideo
        actual_num_frames = video_length  # Use input video_length directly
    
        head_dim = self.model.hidden_size // self.model.heads_num
        rope_dim_list = self.model.rope_dim_list or [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) must equal head_dim"
    
        if actual_num_frames > 192:
            k = 2+((actual_num_frames + 3) // (4 * L_train))
            k = max(4, min(8, k))
            logger.debug(f"actual_num_frames = {actual_num_frames} > 192, RIFLEx applied with k = {k}")
    
            # Compute positional grids for RIFLEx
            axes_grids = [torch.arange(size, device=self.device, dtype=torch.float32) for size in rope_sizes]
            grid = torch.meshgrid(*axes_grids, indexing="ij")
            grid = torch.stack(grid, dim=0)  # [3, t, h, w]
            pos = grid.reshape(3, -1).t()  # [t * h * w, 3]
    
            # Apply RIFLEx to temporal dimension
            freqs = []
            for i in range(3):
                if i == 0:  # Temporal with RIFLEx
                    freqs_cos, freqs_sin = get_1d_rotary_pos_embed_riflex(
                        rope_dim_list[i],
                        pos[:, i],
                        theta=self.args.rope_theta,
                        use_real=True,
                        k=k,
                        L_test=L_test
                    )
                else:  # Spatial with default RoPE
                    freqs_cos, freqs_sin = get_1d_rotary_pos_embed_riflex(
                        rope_dim_list[i],
                        pos[:, i],
                        theta=self.args.rope_theta,
                        use_real=True,
                        k=None,
                        L_test=None
                    )
                freqs.append((freqs_cos, freqs_sin))
                logger.debug(f"freq[{i}] shape: {freqs_cos.shape}, device: {freqs_cos.device}")
    
            freqs_cos = torch.cat([f[0] for f in freqs], dim=1)
            freqs_sin = torch.cat([f[1] for f in freqs], dim=1)
            logger.debug(f"freqs_cos shape: {freqs_cos.shape}, device: {freqs_cos.device}")
        else:
            # 20250316 pftq: Original code for <= 192 frames
            logger.debug(f"actual_num_frames = {actual_num_frames} <= 192, using original RoPE")
            freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
                rope_dim_list,
                rope_sizes,
                theta=self.args.rope_theta,
                use_real=True,
                theta_rescale_factor=1,
            )
            logger.debug(f"freqs_cos shape: {freqs_cos.shape}, device: {freqs_cos.device}")
    
        return freqs_cos, freqs_sin

    @torch.no_grad()
    def predict(
        self,
        prompt,
        height=192,
        width=336,
        video_length=129,
        seed=None,
        negative_prompt=None,
        infer_steps=50,
        guidance_scale=6,
        flow_shift=5.0,
        embedded_guidance_scale=None,
        batch_size=1,
        num_videos_per_prompt=1,
        audio_embeds=None,  # 新增：音频嵌入
        face_embeds=None,   # 新增：人脸嵌入（可选）
        i2v_mode=False,
        i2v_resolution="720p",
        i2v_image_path=None,
        i2v_condition_type=None,
        i2v_stability=True,
        **kwargs,
    ):
        """
        从给定的文本和音频输入预测视频。
        
        Args:
            prompt (str or List[str]): 输入文本提示。
            audio_embeds (torch.Tensor): 音频嵌入张量。
            kwargs:
                height (int): 输出视频的高度。默认为192。
                width (int): 输出视频的宽度。默认为336。
                video_length (int): 输出视频的帧数。默认为129。
                seed (int or List[str]): 生成的随机种子。默认为随机整数。
                negative_prompt (str or List[str]): 负面文本提示。默认为空字符串。
                guidance_scale (float): 生成的引导比例。默认为6.0。
                num_videos_per_prompt (int): 每个提示的视频数量。默认为1。
                infer_steps (int): 推理步骤的数量。默认为50。
                face_embeds (torch.Tensor): 人脸嵌入张量（可选）。
        """
        
        # 基本参数处理与父类相同
        out_dict = dict()

        # 处理种子
        if isinstance(seed, torch.Tensor):
            seed = seed.tolist()
        if seed is None:
            seeds = [
                random.randint(0, 1_000_000)
                for _ in range(batch_size * num_videos_per_prompt)
            ]
        elif isinstance(seed, int):
            seeds = [
                seed + i for _ in range(batch_size)
                for i in range(num_videos_per_prompt)
            ]
        elif isinstance(seed, (list, tuple)):
            if len(seed) == batch_size:
                seeds = [
                    int(seed[i]) + j for i in range(batch_size)
                    for j in range(num_videos_per_prompt)
                ]
            elif len(seed) == batch_size * num_videos_per_prompt:
                seeds = [int(s) for s in seed]
            else:
                raise ValueError(
                    f"Length of seed must be equal to number of prompt(batch_size) or "
                    f"batch_size * num_videos_per_prompt ({batch_size} * {num_videos_per_prompt}), got {seed}."
                )
        else:
            raise ValueError(
                f"Seed must be an integer, a list of integers, or None, got {seed}."
            )
        generator = [
            torch.Generator(self.device).manual_seed(seed) for seed in seeds
        ]
        out_dict["seeds"] = seeds

        # 验证并调整视频尺寸
        if width <= 0 or height <= 0 or video_length <= 0:
            raise ValueError(
                f"`height` and `width` and `video_length` must be positive integers, got height={height}, width={width}, video_length={video_length}"
            )
        if (video_length - 1) % 4 != 0:
            raise ValueError(
                f"`video_length-1` must be a multiple of 4, got {video_length}"
            )

        logger.info(
            f"Input (height, width, video_length) = ({height}, {width}, {video_length})"
        )

        target_height = align_to(height, 16)
        target_width = align_to(width, 16)
        target_video_length = video_length

        out_dict["size"] = (target_height, target_width, target_video_length)

        # 处理提示词
        if not isinstance(prompt, str):
            raise TypeError(
                f"`prompt` must be a string, but got {type(prompt)}")
        prompt = [prompt.strip()]

        # 负面提示词
        if negative_prompt is None or negative_prompt == "":
            negative_prompt = self.default_negative_prompt
        if not isinstance(negative_prompt, str):
            raise TypeError(
                f"`negative_prompt` must be a string, but got {type(negative_prompt)}"
            )
        negative_prompt = [negative_prompt.strip()]

        # 设置调度器
        scheduler = FlowMatchDiscreteScheduler(
            shift=flow_shift,
            reverse=self.args.flow_reverse,
            solver=self.args.flow_solver,
        )
        self.pipeline.scheduler = scheduler

        img_latents = None
        semantic_images = None
        if i2v_mode:
            if i2v_resolution == "720p":
                bucket_hw_base_size = 960
            elif i2v_resolution == "540p":
                bucket_hw_base_size = 720
            elif i2v_resolution == "360p":
                bucket_hw_base_size = 480
            else:
                raise ValueError(f"i2v_resolution: {i2v_resolution} must be in [360p, 540p, 720p]")

            semantic_images = [Image.open(i2v_image_path).convert('RGB')]
            origin_size = semantic_images[0].size

            crop_size_list = generate_crop_size_list(bucket_hw_base_size, 32)
            aspect_ratios = np.array([round(float(h)/float(w), 5) for h, w in crop_size_list])
            closest_size, closest_ratio = get_closest_ratio(origin_size[1], origin_size[0], aspect_ratios, crop_size_list)


            resize_param = min(closest_size)
            center_crop_param = closest_size

            ref_image_transform = transforms.Compose([
                transforms.Resize(resize_param),
                transforms.CenterCrop(center_crop_param),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

            semantic_image_pixel_values = [ref_image_transform(semantic_image) for semantic_image in semantic_images]
            semantic_image_pixel_values = torch.cat(semantic_image_pixel_values).unsqueeze(0).unsqueeze(2).to(self.device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                img_latents = self.pipeline.vae.encode(semantic_image_pixel_values).latent_dist.mode()
                img_latents.mul_(self.pipeline.vae.config.scaling_factor)

            target_height, target_width = closest_size

        freqs_cos, freqs_sin = self.get_rotary_pos_embed(
            target_video_length, target_height, target_width
        )

        n_tokens = freqs_cos.shape[0]

        # 打印推理参数
        debug_str = f"""
                        height: {target_height}
                         width: {target_width}
                  video_length: {target_video_length}
                        prompt: {prompt}
                    neg_prompt: {negative_prompt}
                          seed: {seed}
                   infer_steps: {infer_steps}
         num_videos_per_prompt: {num_videos_per_prompt}
                guidance_scale: {guidance_scale}
                      n_tokens: {n_tokens}
                    flow_shift: {flow_shift}
       embedded_guidance_scale: {embedded_guidance_scale}
               audio_embeds: {"有" if audio_embeds is not None else "无"}
                face_embeds: {"有" if face_embeds is not None else "无"}"""
        logger.debug(debug_str)

        # 验证音频嵌入
        if audio_embeds is not None:
            logger.info(f"音频嵌入形状: {audio_embeds.shape}")
            # 确保音频嵌入与视频长度兼容
            if hasattr(audio_embeds, "shape") and len(audio_embeds.shape) >= 2:
                audio_frames = audio_embeds.shape[1]  # 假设形状为[batch, frames, ...]
                if audio_frames != target_video_length:
                    logger.warning(f"音频帧数({audio_frames})与目标视频帧数({target_video_length})不匹配")

        # Pipeline推理

        start_time = time.time()
        samples = self.pipeline(
            prompt=prompt,
            height=target_height,
            width=target_width,
            video_length=target_video_length,
            num_inference_steps=infer_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            output_type="pil",
            freqs_cis=(freqs_cos, freqs_sin),
            n_tokens=n_tokens,
            embedded_guidance_scale=embedded_guidance_scale,
            data_type="video" if target_video_length > 1 else "image",
            is_progress_bar=True,
            vae_ver=self.args.vae,
            enable_tiling=self.args.vae_tiling,
            enable_vae_sp=self.args.vae_sp,
            audio_embeds=audio_embeds,  # 传递音频嵌入
            face_embeds=face_embeds,    # 传递人脸嵌入
            i2v_mode=i2v_mode,
            i2v_condition_type=i2v_condition_type,
            i2v_stability=i2v_stability,
            img_latents=img_latents,
            semantic_images=semantic_images,
        )[0]
        out_dict["samples"] = samples
        out_dict["prompts"] = prompt

        gen_time = time.time() - start_time
        logger.info(f"Success, time: {gen_time}")

        return out_dict
