import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from loguru import logger
from safetensors.torch import load_file as safetensors_load_file

from fastvideo.models.hunyuan.constants import (NEGATIVE_PROMPT,
                                                PRECISION_TO_TYPE,
                                                PROMPT_TEMPLATE)
from fastvideo.models.hunyuan.diffusion.pipelines import HunyuanVideoPipeline
from fastvideo.models.hunyuan.diffusion.pipelines.pipeline_hunyuan_video import HunyuanVideoAudioPipeline
from fastvideo.models.hunyuan.diffusion.schedulers import \
    FlowMatchDiscreteScheduler
from fastvideo.models.hunyuan.modules import load_model
from fastvideo.models.hunyuan.text_encoder import TextEncoder
from fastvideo.models.hunyuan.utils.data_utils import align_to
from fastvideo.models.hunyuan.vae import load_vae
from fastvideo.utils.parallel_states import nccl_info


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
            device (int): The device for inference. Default is 0.
        """
        # ========================================================================
        logger.info(
            f"Got text-to-video model root path: {pretrained_model_path}")

        # ==================== Initialize Distributed Environment ================
        # Ensure distributed is initialized if needed for loading sharded checkpoints
        if args.sp_size > 1 and not dist.is_initialized():
             # This assumes init_process_group is called elsewhere before from_pretrained
             # If not, it needs to be initialized here or earlier.
             # For now, we assume it's initialized if sp_size > 1
             if os.environ.get("RANK") is not None and os.environ.get("WORLD_SIZE") is not None:
                 logger.warning("Distributed environment not initialized, but SP size > 1. Assuming it's handled externally.")
             # else:
                 # raise RuntimeError("Distributed environment must be initialized for SP > 1.")

        if nccl_info.sp_size > 1:
            device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        parallel_args = None  # {"ulysses_degree": args.ulysses_degree, "ring_degree": args.ring_degree}

        # ======================== Get the args path =============================

        # Disable gradient
        torch.set_grad_enabled(False)

        # =========================== Build main model ===========================
        logger.info("Building model...")
        factor_kwargs = {
            "device": device,
            "dtype": PRECISION_TO_TYPE[args.precision]
        }
        in_channels = args.latent_channels
        out_channels = args.latent_channels

        model = load_model(
            args,
            in_channels=in_channels,
            out_channels=out_channels,
            factor_kwargs=factor_kwargs,
        )

        # --- Crucial Change: Move model loading AFTER potential FSDP wrapping ---
        # The original code loaded state_dict before potential FSDP wrapping.
        # If loading sharded state, the model MUST be FSDP wrapped first.
        # We will load the state dict later.
        # For now, just move the model to the correct device.
        model = model.to(device)

        # --- Potential FSDP Wrapping (if applicable, based on your training setup) ---
        # If your inference needs FSDP wrapping (e.g., for very large models),
        # you would wrap the model here *before* loading the state dict.
        # Example (needs correct FSDP config):
        # if args.use_fsdp_inference: # Add an argument like this if needed
        #     fsdp_config = {...} # Your FSDP configuration
        #     model = FSDP(model, **fsdp_config)
        #     dist.barrier() # Ensure all ranks wrap before loading

        # --- Load State Dict (Now potentially on an FSDP model) ---
        model = Inference.load_state_dict(args, model, pretrained_model_path, device)
        model.eval()


        # ============================= Build extra models ========================
        # VAE
        vae, _, s_ratio, t_ratio = load_vae(
            args.vae,
            args.vae_precision,
            logger=logger,
            device=device if not args.use_cpu_offload else "cpu",
        )
        vae_kwargs = {"s_ratio": s_ratio, "t_ratio": t_ratio}

        # Text encoder
        if args.prompt_template_video is not None:
            crop_start = PROMPT_TEMPLATE[args.prompt_template_video].get(
                "crop_start", 0)
        elif args.prompt_template is not None:
            crop_start = PROMPT_TEMPLATE[args.prompt_template].get(
                "crop_start", 0)
        else:
            crop_start = 0
        max_length = args.text_len + crop_start

        # prompt_template
        prompt_template = (PROMPT_TEMPLATE[args.prompt_template]
                           if args.prompt_template is not None else None)

        # prompt_template_video
        prompt_template_video = (PROMPT_TEMPLATE[args.prompt_template_video]
                                 if args.prompt_template_video is not None else
                                 None)

        text_encoder = TextEncoder(
            text_encoder_type=args.text_encoder,
            max_length=max_length,
            text_encoder_precision=args.text_encoder_precision,
            tokenizer_type=args.tokenizer,
            prompt_template=prompt_template,
            prompt_template_video=prompt_template_video,
            hidden_state_skip_layer=args.hidden_state_skip_layer,
            apply_final_norm=args.apply_final_norm,
            reproduce=args.reproduce,
            logger=logger,
            device=device if not args.use_cpu_offload else "cpu",
        )
        text_encoder_2 = None
        if args.text_encoder_2 is not None:
            text_encoder_2 = TextEncoder(
                text_encoder_type=args.text_encoder_2,
                max_length=args.text_len_2,
                text_encoder_precision=args.text_encoder_precision_2,
                tokenizer_type=args.tokenizer_2,
                reproduce=args.reproduce,
                logger=logger,
                device=device if not args.use_cpu_offload else "cpu",
            )

        return cls(
            args=args,
            vae=vae,
            vae_kwargs=vae_kwargs,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            model=model,
            use_cpu_offload=args.use_cpu_offload,
            device=device,
            logger=logger,
            parallel_args=parallel_args,
        )

    @staticmethod
    def load_state_dict(args, model, pretrained_model_path, device):
        """
        Loads the model state dict. Supports loading single files (.pt, .safetensors)
        and FSDP sharded checkpoints saved via dist_cp.

        Args:
            args: Command line arguments. Expected to have `dit_weight` and potentially others.
            model: The model instance (potentially FSDP wrapped).
            pretrained_model_path: Base path for models if `dit_weight` is relative or not provided.
            device: The target device.

        Returns:
            The model with loaded state dict.
        """
        load_key = args.load_key # Used for non-sharded legacy checkpoints
        dit_weight_path = Path(args.dit_weight) if args.dit_weight else None

        # Determine the final path to load from
        load_path = None
        is_sharded = False

        if dit_weight_path:
            if dit_weight_path.is_dir():
                # Check if it's a sharded checkpoint directory
                model_state_dir = dit_weight_path / "model_sharded_state"
                if model_state_dir.is_dir():
                    logger.info(f"Detected FSDP sharded checkpoint directory: {dit_weight_path}")
                    load_path = dit_weight_path
                    is_sharded = True
                else:
                    # Assume it's a directory containing single weight files (legacy)
                    logger.warning(f"Directory specified in --dit-weight ({dit_weight_path}) does not contain 'model_sharded_state'. Looking for legacy files...")
                    files = list(dit_weight_path.glob("*.pt")) + list(dit_weight_path.glob("*.safetensors"))
                    if not files:
                        raise ValueError(f"No model weights (.pt or .safetensors) found in directory: {dit_weight_path}")
                    # Attempt to find standard names, default to the first found file otherwise
                    potential_paths = [
                        dit_weight_path / f"pytorch_model_{load_key}.pt",
                        dit_weight_path / "diffusion_pytorch_model.safetensors", # Common name
                        dit_weight_path / "model.safetensors"
                    ]
                    found = False
                    for p in potential_paths:
                        if p.is_file():
                            load_path = p
                            found = True
                            logger.info(f"Found legacy weight file: {load_path}")
                            break
                    if not found:
                        load_path = files[0] # Fallback to the first file found
                        logger.warning(f"Could not find standard weight names, loading first found file: {load_path}")

            elif dit_weight_path.is_file():
                # It's a direct path to a single weight file
                load_path = dit_weight_path
                logger.info(f"Loading single weight file: {load_path}")
            else:
                raise FileNotFoundError(f"Specified --dit-weight path does not exist: {dit_weight_path}")
        else:
            # --dit-weight not provided, fall back to looking in pretrained_model_path (legacy behavior)
            logger.warning("--dit-weight not provided. Attempting to find weights in pretrained_model_path (legacy behavior).")
            model_dir = Path(pretrained_model_path) / f"t2v_{args.model_resolution}" # Example legacy structure
            files = list(model_dir.glob("*.pt")) + list(model_dir.glob("*.safetensors"))
            if not files:
                raise ValueError(f"No model weights found in legacy path: {model_dir}")

            potential_paths = [
                model_dir / f"pytorch_model_{load_key}.pt",
                model_dir / "diffusion_pytorch_model.safetensors",
            ]
            found = False
            for p in potential_paths:
                if p.is_file():
                    load_path = p
                    found = True
                    logger.info(f"Found legacy weight file: {load_path}")
                    break
            if not found:
                load_path = files[0] # Fallback
                logger.warning(f"Could not find standard legacy weight names in {model_dir}, loading first found file: {load_path}")


        if load_path is None:
            raise ValueError("Could not determine a valid model weight path to load.")

        # --- Loading Logic ---
        if is_sharded:
            # Load FSDP Sharded Checkpoint
            model_state_dir = load_path / "model_sharded_state"
            logger.info(f"Loading FSDP sharded model state from: {model_state_dir}")

            if not isinstance(model, FSDP):
                 logger.warning("Loading a sharded checkpoint, but the model is not wrapped with FSDP. This might lead to errors or unexpected behavior. Consider wrapping the model with FSDP during initialization if necessary.")
                 # Attempt loading anyway, but it might fail depending on the model structure

            # Use the SHARDED_STATE_DICT context manager for loading
            try:
                with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                    # Create a dictionary matching the saved structure
                    model_state_dict_to_load = {"model": model.state_dict()}

                    dist_cp.load_state_dict(
                        state_dict=model_state_dict_to_load,
                        storage_reader=dist_cp.FileSystemReader(str(model_state_dir)),
                        planner=DefaultLoadPlanner(),
                        no_dist=False # Perform distributed loading
                    )

                    # Apply the loaded sharded state to the model
                    model.load_state_dict(model_state_dict_to_load["model"])
                    logger.info(f"Successfully loaded sharded model state from {model_state_dir}")

            except Exception as e:
                logger.error(f"Failed to load sharded state dict from {model_state_dir}: {e}")
                # If FSDP context fails (e.g., model not wrapped), try a direct load approach (less safe)
                logger.warning("Trying direct state dict loading as fallback (may be incorrect for FSDP)...")
                try:
                    # This is less standard for sharded checkpoints but might work in some cases
                    state_dict = {"model": model.state_dict()} # Get current structure
                    dist_cp.load_state_dict(
                        state_dict=state_dict,
                        storage_reader=dist_cp.FileSystemReader(str(model_state_dir))
                        )
                    model.load_state_dict(state_dict["model"]) # Load flattened state directly
                    logger.info("Successfully loaded sharded state dict using fallback.")
                except Exception as fallback_e:
                     logger.error(f"Fallback loading also failed: {fallback_e}")
                     raise RuntimeError(f"Could not load sharded checkpoint from {model_state_dir}") from e

            # Barrier to ensure all ranks finish loading before proceeding
            if dist.is_initialized():
                dist.barrier()

        else:
            # Load Single File Checkpoint (.pt or .safetensors)
            logger.info(f"Loading single-file model state from: {load_path}...")
            if load_path.suffix == ".safetensors":
                state_dict = safetensors_load_file(str(load_path), device=str(device)) # Load directly to target device if possible
            elif load_path.suffix == ".pt":
                state_dict = torch.load(str(load_path), map_location=device) # map_location handles device placement
            else:
                raise ValueError(f"Unsupported file format: {load_path}")

            # Handle potential nesting (e.g., {'module': ..., 'ema': ...})
            bare_model = True
            if isinstance(state_dict, dict) and ("module" in state_dict or "ema" in state_dict or load_key in state_dict):
                if load_key in state_dict:
                    logger.info(f"Extracting state dict with key '{load_key}'")
                    state_dict = state_dict[load_key]
                    bare_model = False
                elif "module" in state_dict: # Common key from DDP or FSDP saves
                     logger.warning(f"Key '{load_key}' not found, but 'module' key exists. Using 'module' state dict.")
                     state_dict = state_dict["module"]
                     bare_model = False
                # Add more potential keys if needed

            # Check if model is FSDP wrapped, state_dict might need unwrapping if saved from FSDP FullStateDict
            if isinstance(model, FSDP) and not bare_model:
                 logger.warning("Model is FSDP wrapped, but loading a potentially nested single-file checkpoint. Ensure the state_dict keys match the FSDP model parameters.")
                 # FSDP's load_state_dict might handle this automatically if keys match flattened params.

            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logger.warning(f"Missing keys during state_dict load: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys during state_dict load: {unexpected_keys}")
            logger.info(f"Successfully loaded single-file model state from {load_path}")

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


class HunyuanVideoSampler(Inference):

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
        """Load the denoising scheduler for inference."""
        if scheduler is None:
            if args.denoise_type == "flow":
                scheduler = FlowMatchDiscreteScheduler(
                    shift=args.flow_shift,
                    reverse=args.flow_reverse,
                    solver=args.flow_solver,
                )
            else:
                raise ValueError(f"Invalid denoise type {args.denoise_type}")

        pipeline = HunyuanVideoPipeline(
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
        **kwargs,
    ):
        """
        Predict the image/video from the given text.

        Args:
            prompt (str or List[str]): The input text.
            kwargs:
                height (int): The height of the output video. Default is 192.
                width (int): The width of the output video. Default is 336.
                video_length (int): The frame number of the output video. Default is 129.
                seed (int or List[str]): The random seed for the generation. Default is a random integer.
                negative_prompt (str or List[str]): The negative text prompt. Default is an empty string.
                guidance_scale (float): The guidance scale for the generation. Default is 6.0.
                num_images_per_prompt (int): The number of images per prompt. Default is 1.
                infer_steps (int): The number of inference steps. Default is 100.
        """

        out_dict = dict()

        # ========================================================================
        # Arguments: seed
        # ========================================================================
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
        # Peiyuan: using GPU seed will cause A100 and H100 to generate different results...
        generator = [
            torch.Generator("cpu").manual_seed(seed) for seed in seeds
        ]
        out_dict["seeds"] = seeds

        # ========================================================================
        # Arguments: target_width, target_height, target_video_length
        # ========================================================================
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

        # ========================================================================
        # Arguments: prompt, new_prompt, negative_prompt
        # ========================================================================
        if not isinstance(prompt, str):
            raise TypeError(
                f"`prompt` must be a string, but got {type(prompt)}")
        prompt = [prompt.strip()]

        # negative prompt
        if negative_prompt is None or negative_prompt == "":
            negative_prompt = self.default_negative_prompt
        if not isinstance(negative_prompt, str):
            raise TypeError(
                f"`negative_prompt` must be a string, but got {type(negative_prompt)}"
            )
        negative_prompt = [negative_prompt.strip()]

        # ========================================================================
        # Scheduler
        # ========================================================================
        scheduler = FlowMatchDiscreteScheduler(
            shift=flow_shift,
            reverse=self.args.flow_reverse,
            solver=self.args.flow_solver,
        )
        self.pipeline.scheduler = scheduler

        if "884" in self.args.vae:
            latents_size = [(video_length - 1) // 4 + 1, height // 8,
                            width // 8]
        elif "888" in self.args.vae:
            latents_size = [(video_length - 1) // 8 + 1, height // 8,
                            width // 8]
        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

        # ========================================================================
        # Print infer args
        # ========================================================================
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
       embedded_guidance_scale: {embedded_guidance_scale}"""
        logger.debug(debug_str)

        # ========================================================================
        # Pipeline inference
        # ========================================================================
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
            n_tokens=n_tokens,
            embedded_guidance_scale=embedded_guidance_scale,
            data_type="video" if target_video_length > 1 else "image",
            is_progress_bar=True,
            vae_ver=self.args.vae,
            enable_tiling=self.args.vae_tiling,
            enable_vae_sp=self.args.vae_sp,
        )[0]
        out_dict["samples"] = samples
        out_dict["prompts"] = prompt

        gen_time = time.time() - start_time
        logger.info(f"Success, time: {gen_time}")

        return out_dict


class HunyuanAudioVideoSampler(HunyuanVideoSampler):
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

        # 注意这里使用HunyuanVideoAudioPipeline替代了HunyuanVideoPipeline
        pipeline = HunyuanVideoAudioPipeline(
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
            torch.Generator("cpu").manual_seed(seed) for seed in seeds
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

        if "884" in self.args.vae:
            latents_size = [(video_length - 1) // 4 + 1, height // 8,
                            width // 8]
        elif "888" in self.args.vae:
            latents_size = [(video_length - 1) // 8 + 1, height // 8,
                            width // 8]
        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

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
            n_tokens=n_tokens,
            embedded_guidance_scale=embedded_guidance_scale,
            data_type="video" if target_video_length > 1 else "image",
            is_progress_bar=True,
            vae_ver=self.args.vae,
            enable_tiling=self.args.vae_tiling,
            enable_vae_sp=self.args.vae_sp,
            audio_embeds=audio_embeds,  # 传递音频嵌入
            face_embeds=face_embeds,    # 传递人脸嵌入
        )[0]
        out_dict["samples"] = samples
        out_dict["prompts"] = prompt

        gen_time = time.time() - start_time
        logger.info(f"Success, time: {gen_time}")

        return out_dict
