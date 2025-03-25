export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export WANDB_API_KEY="5b140725d3c02629d4f7599685125eb24df88b79"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 根据CUDA_VISIBLE_DEVICES设置sp_size和nproc_per_node
nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
sp_size=1

# echo "启动训练，只训练音频相关参数..."

torchrun --nnodes 1 --nproc_per_node $nproc_per_node \
    fastvideo/train_audio.py \
    --seed 42 \
    --pretrained_model_name_or_path /sds_wangby/models/HunyuanVideo/HunyuanVideo \
    --dit_model_name_or_path /data/nas/yexin/workspace/shunian/model/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt\
    --model_type "hunyuan_audio" \
    --cache_dir data/.cache \
    --data_json_path data/50_hour_test_480p_49frames/videos2caption.json \
    --validation_prompt_dir data/Image-Vid-Finetune-HunYuan/validation \
    --gradient_checkpointing \
    --train_batch_size=4 \
    --num_latent_t 32 \
    --sp_size $sp_size \
    --train_sp_batch_size 4 \
    --dataloader_num_workers 8 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=3000 \
    --learning_rate=2e-5 \
    --mixed_precision=bf16 \
    --checkpointing_steps=100 \
    --validation_steps 3000 \
    --validation_sampling_steps 50 \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.0 \
    --ema_decay 0.999 \
    --log_validation \
    --output_dir=data/outputs/Hunyuan-Audio-Finetune-Hunyuan-audio-only-49frames \
    --tracker_project_name Hunyuan-Audio-Finetune-Hunyuan \
    --num_frames 49 \
    --num_height 480 \
    --num_width 848 \
    --shift 7 \
    --validation_guidance_scale "1.0" \
    --master_weight_type bf16 \
    --train_audio_only \
    > ./logs/finetune_hunyuan_audio.log 2>&1