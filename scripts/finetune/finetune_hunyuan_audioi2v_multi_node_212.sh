export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export WANDB_API_KEY="5b140725d3c02629d4f7599685125eb24df88b79"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


export NCCL_DEBUG=INFO


# 根据CUDA_VISIBLE_DEVICES设置sp_size和nproc_per_node
nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
sp_size=1
IP=172.27.38.139

DATA_NAME="200_hour_480p_49frames_eng"
EXPERIMENT_NAME="Hunyuan-Audio-Finetune-Hunyuan-ai2v-49frames-${DATA_NAME}-0503-debug-yexin3"
LOG_DIR="/data/nas/yexin/workspace/shunian/model_training/FastVideo/logs/model_training/${EXPERIMENT_NAME}_212_debug_yexin2.log"
OUTPUT_DIR="./data/outputs/${EXPERIMENT_NAME}"
mkdir -p $(dirname $LOG_DIR)
mkdir -p $(dirname $OUTPUT_DIR)

te_params=" \
    --vae-model-name-or-path /data/nas/yexin/workspace/shunian/model/ \
    --vae-precision fp16 \
    --text-encoder llm-i2v \
    --text-encoder-precision fp16 \
    --text-states-dim 4096 \
    --text-len 256 \
    --tokenizer llm-i2v \
    --prompt-template dit-llm-encode-i2v \
    --prompt-template-video dit-llm-encode-video-i2v \
    --hidden-state-skip-layer 2 \
    --text-encoder-2 clipL \
    --text-encoder-precision-2 fp16 \
    --text-states-dim-2 768 \
    --tokenizer-2 clipL \
    --text-len-2 77 \
    --i2v-mode \
    --reproduce \
    "

# echo "启动训练，只训练音频相关参数..."

export MODEL_BASE="/data/nas/yexin/workspace/shunian/model"

torchrun --nnodes 2 --nproc_per_node $nproc_per_node \
    --node_rank=0 \
    --rdzv_id=456 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$IP:29500 \
    fastvideo/train_audio_i2v.py \
    --seed 42 \
    --pretrained_model_name_or_path /sds_wangby/models/HunyuanVideo/HunyuanVideo \
    --dit_model_name_or_path ${MODEL_BASE}/hunyuan-video-i2v-720p/transformers/mp_rank_00_model_states.pt\
    --model_type "hunyuan_audio_i2v" \
    --cache_dir data/.cache \
    --data_json_path data/${DATA_NAME}/videos2caption_cleaned.json \
    --validation_prompt_dir data/Image-Vid-Finetune-HunYuan/validation \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --num_latent_t 32 \
    --sp_size $sp_size \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 8 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=6000 \
    --learning_rate=1e-5 \
    --mixed_precision=bf16 \
    --checkpointing_steps=50 \
    --validation_steps 6000 \
    --validation_sampling_steps 50 \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0 \
    --ema_decay 0.999 \
    --log_validation \
    --output_dir=${OUTPUT_DIR} \
    --tracker_project_name Hunyuan-Audio-Finetune-Hunyuan \
    --num_frames 49 \
    --num_height 480 \
    --num_width 848 \
    --shift 7 \
    --validation_guidance_scale "1.0" \
    --master_weight_type bf16 \
    --train_audio_only \
    --resume_from_checkpoint "auto" \
    ${te_params} \
    > ${LOG_DIR} 2>&1