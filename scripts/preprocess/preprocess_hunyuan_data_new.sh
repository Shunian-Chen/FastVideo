# export WANDB_MODE="offline"
# GPU_ID
GPU_ID=0,1,2,3
export CUDA_VISIBLE_DEVICES=$GPU_ID
# 根据GPU_ID设置GPU_NUM
GPU_NUM=$(echo $GPU_ID | tr ',' '\n' | wc -l)
MODEL_PATH="/data/nas/yexin/workspace/shunian/model"
MODEL_TYPE="hunyuan"

# 定义 prepare_preprocess_data.py 所需的路径
# !! 请根据你的实际情况修改这些路径 !!
PREPARE_ORIGINAL_DATA_PATH="/data/nas/yexin/workspace/shunian/data/0_30000_fps24_121frames_uniform_distribution_in_label_combinations2.json" # 例如: data/50_hour_test/metadata.json
PREPARE_CAPTION_DATA_PATH="/data/nas/yexin/workspace/shunian/model_training/FastVideo/data/252_hour_test_480p_49frames/0_30000_fps24_121frames_all_corrected_with_descriptions.json" # 如果有的话，否则留空或不传此参数
PREPARE_ORIGINAL_VIDEO_DIR="/sds_wangby/datasets_dir/datasets/shunian/workspace/talking_face/Talking-Face-Datapipe/outputs/common/" # 例如: /datasets/shunian/raw_videos
PREPARE_TARGET_VIDEO_DIR="/data/nas/yexin/workspace/shunian/data/" # 例如: /data/nas/yexin/workspace/shunian/data/
num_frames=121


# DATA_MERGE_PATH="data/50_hour_test/merge.txt" # 这个路径现在可能由 prepare_preprocess_data.py 生成的 data.json 取代
# OUTPUT_DIR="data/252_hour_test_480p_${num_frames}frames" # 这个目录将用于 prepare_preprocess_data.py 和后续步骤
# OUTPUT_DIR="data/test"
OUTPUT_DIR="data/evaluation_480p_${num_frames}frames"
DATA_FOR_PREPROCESS_PATH="${OUTPUT_DIR}/data.json" # 更新 DATA_MERGE_PATH 指向 prepare_preprocess_data.py 的输出
DATA_MERGE_PATH="${OUTPUT_DIR}/merge.txt"
LOG_DIR="logs/${OUTPUT_DIR}"
VALIDATION_PATH="assets/prompt.txt"

# 创建log目录
mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR # 确保输出目录存在

# # 设置NCCL环境变量以提高稳定性
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
# 移除网络接口设置，让NCCL自动选择
# export NCCL_SOCKET_IFNAME=eth0

echo "DATA_FOR_PREPROCESS_PATH: $DATA_FOR_PREPROCESS_PATH"
MASTER_PORT=29502



# # 1. 运行 prepare_preprocess_data.py
# echo "Running prepare_preprocess_data.py..."
# python fastvideo/data_preprocess/prepare_preprocess_data.py \
#     --original_data_path $PREPARE_ORIGINAL_DATA_PATH \
#     --output_dir $OUTPUT_DIR \
#     --original_video_dir $PREPARE_ORIGINAL_VIDEO_DIR \
#     --target_video_dir $PREPARE_TARGET_VIDEO_DIR \
#     ${PREPARE_CAPTION_DATA_PATH:+--caption_data_path $PREPARE_CAPTION_DATA_PATH} \
#     > ./$LOG_DIR/prepare_preprocess_data.log 2>&1
# echo "Finished prepare_preprocess_data.py."

# touch $DATA_MERGE_PATH
# # 将DATA_FOR_PREPROCESS_PATH中的数据写入DATA_MERGE_PATH
# echo "$OUTPUT_DIR/video,$DATA_FOR_PREPROCESS_PATH" > $DATA_MERGE_PATH


# 2. 运行 preprocess_vae_latents_origin.py
# 确保每个进程使用不同的GPU
echo "Running preprocess_vae_latents_origin.py..."
CUDA_VISIBLE_DEVICES=$GPU_ID torchrun --nproc_per_node=$GPU_NUM --master_port=$MASTER_PORT \
    fastvideo/data_preprocess/preprocess_vae_latents_origin.py \
    --model_path $MODEL_PATH \
    --data_merge_path $DATA_MERGE_PATH \
    --train_batch_size=1 \
    --max_height=480 \
    --max_width=848 \
    --num_frames=$num_frames \
    --dataloader_num_workers 0 \
    --output_dir=$OUTPUT_DIR \
    --model_type $MODEL_TYPE \
    --train_fps 24 \
    > ./$LOG_DIR/process_vae_latents_origin.log 2>&1
echo "Finished preprocess_vae_latents_origin.py."

echo "Running preprocess_text_embeddings.py..."
CUDA_VISIBLE_DEVICES=$GPU_ID torchrun --nproc_per_node=$GPU_NUM --master_port=$MASTER_PORT \
    fastvideo/data_preprocess/preprocess_text_embeddings.py \
    --model_type $MODEL_TYPE \
    --model_path $MODEL_PATH \
    --output_dir=$OUTPUT_DIR \
    > ./$LOG_DIR/process_text_embs.log 2>&1
echo "Finished preprocess_text_embeddings.py."

# bash scripts/finetune/finetune_hunyuan_audio.sh
# torchrun --nproc_per_node=1 \
#     fastvideo/data_preprocess/preprocess_validation_text_embeddings.py \
#     --model_type $MODEL_TYPE \
#     --model_path $MODEL_PATH \
#     --output_dir=$OUTPUT_DIR \
#     --validation_prompt_txt $VALIDATION_PATH