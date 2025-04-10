export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export WANDB_API_KEY="5b140725d3c02629d4f7599685125eb24df88b79"
export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun --nnodes 1 --nproc_per_node 4 \
    fastvideo/test_normalize.py \
    --seed 42 \
    --pretrained_model_name_or_path /sds_wangby/models/HunyuanVideo/HunyuanVideo \
    --dit_model_name_or_path /sds_wangby/models/HunyuanVideo/HunyuanVideo/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt\
    --model_type "hunyuan_audio" \
    --cache_dir data/.cache \
    --data_json_path data/hallo3-data-origin-1k/videos2caption.json \
    --validation_prompt_dir data/Image-Vid-Finetune-HunYuan/validation \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --num_latent_t 32 \
    --sp_size 4 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1 \
    --learning_rate=1e-5 \
    --mixed_precision=bf16 \
    --checkpointing_steps=200 \
    --validation_steps 2000 \
    --validation_sampling_steps 50 \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.0 \
    --ema_decay 0.999 \
    --log_validation \
    --output_dir=data/outputs/HSH-Taylor-Finetune-Hunyuan \
    --tracker_project_name HSH-Taylor-Finetune-Hunyuan \
    --num_frames 93 \
    --num_height 480 \
    --num_width 848 \
    --shift 7 \
    --validation_guidance_scale "1.0" \