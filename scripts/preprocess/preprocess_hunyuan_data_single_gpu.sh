# export WANDB_MODE="offline"
GPU_NUM=1 # 使用单GPU
MODEL_PATH="data/hunyuan"
MODEL_TYPE="hunyuan"

DATA_MERGE_PATH="data/Image-Vid-Finetune-Src/merge.txt"
OUTPUT_DIR="data/Image-Vid-Finetune-HunYuan"
VALIDATION_PATH="assets/prompt.txt"

# 设置NCCL环境变量以提高稳定性
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
# 移除网络接口设置，让NCCL自动选择
# export NCCL_SOCKET_IFNAME=eth0

echo "DATA_MERGE_PATH: $DATA_MERGE_PATH"
MASTER_PORT=29500

# 使用单GPU
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=$GPU_NUM --master_port=$MASTER_PORT \
    fastvideo/data_preprocess/preprocess_vae_latents_origin.py \
    --model_path $MODEL_PATH \
    --data_merge_path $DATA_MERGE_PATH \
    --train_batch_size=1 \
    --max_height=480 \
    --max_width=848 \
    --num_frames=93 \
    --dataloader_num_workers 0 \
    --output_dir=$OUTPUT_DIR \
    --model_type $MODEL_TYPE \
    --train_fps 24 \
    > ./user/logs/process_vae_latents_origin_single_gpu.log 2>&1

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=$GPU_NUM --master_port=$MASTER_PORT \
    fastvideo/data_preprocess/preprocess_text_embeddings.py \
    --model_type $MODEL_TYPE \
    --model_path $MODEL_PATH \
    --output_dir=$OUTPUT_DIR \
    > ./user/logs/process_text_embs_single_gpu.log 2>&1

# torchrun --nproc_per_node=1 \
#     fastvideo/data_preprocess/preprocess_validation_text_embeddings.py \
#     --model_type $MODEL_TYPE \
#     --model_path $MODEL_PATH \
#     --output_dir=$OUTPUT_DIR \
#     --validation_prompt_txt $VALIDATION_PATH 