# export WANDB_MODE="offline"
GPU_NUM=8 # 2,4,8
MODEL_PATH="/data/nas/yexin/workspace/shunian/model"
MODEL_TYPE="hunyuan"

DATA_MERGE_PATH="data/50_hour_test/merge.txt"
OUTPUT_DIR="data/50_hour_test_480p_49frames"
LOG_DIR="logs/${OUTPUT_DIR}"
VALIDATION_PATH="assets/prompt.txt"

# 创建log目录
mkdir -p $LOG_DIR

# # 设置NCCL环境变量以提高稳定性
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
# 移除网络接口设置，让NCCL自动选择
# export NCCL_SOCKET_IFNAME=eth0

echo "DATA_MERGE_PATH: $DATA_MERGE_PATH"
MASTER_PORT=29500

# 根据GPU_NUM设置GPU_ID
GPU_ID=$(seq -s, 0 $((GPU_NUM-1)))

# 确保每个进程使用不同的GPU
CUDA_VISIBLE_DEVICES=$GPU_ID torchrun --nproc_per_node=$GPU_NUM --master_port=$MASTER_PORT \
    fastvideo/data_preprocess/preprocess_vae_latents_origin.py \
    --model_path $MODEL_PATH \
    --data_merge_path $DATA_MERGE_PATH \
    --train_batch_size=1 \
    --max_height=480 \
    --max_width=848 \
    --num_frames=49 \
    --dataloader_num_workers 0 \
    --output_dir=$OUTPUT_DIR \
    --model_type $MODEL_TYPE \
    --train_fps 24 \
    > ./$LOG_DIR/process_vae_latents_origin.log 2>&1

CUDA_VISIBLE_DEVICES=$GPU_ID torchrun --nproc_per_node=$GPU_NUM --master_port=$MASTER_PORT \
    fastvideo/data_preprocess/preprocess_text_embeddings.py \
    --model_type $MODEL_TYPE \
    --model_path $MODEL_PATH \
    --output_dir=$OUTPUT_DIR \
    > ./$LOG_DIR/process_text_embs.log 2>&1


# bash scripts/finetune/finetune_hunyuan_audio.sh
# torchrun --nproc_per_node=1 \
#     fastvideo/data_preprocess/preprocess_validation_text_embeddings.py \
#     --model_type $MODEL_TYPE \
#     --model_path $MODEL_PATH \
#     --output_dir=$OUTPUT_DIR \
#     --validation_prompt_txt $VALIDATION_PATH