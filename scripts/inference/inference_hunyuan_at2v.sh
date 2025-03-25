#!/bin/bash

# 设置GPU数量，默认为1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=${1:-8}
# 设置主端口
MASTER_PORT=${2:-29505}

echo "NUM_GPUS: ${NUM_GPUS}"
echo "MASTER_PORT: ${MASTER_PORT}"

# 数据目录结构
# - videos2caption.json: 包含视频标题与音频嵌入文件路径的JSON文件
# - audio_emb/: 存放音频嵌入文件 (.pt)
# - audios/: 存放对应的音频文件 (.wav)
# 数据目录
DATA_DIR="/data/nas/yexin/workspace/shunian/model_training/FastVideo/data/50_hour_test_480p_49frames"
INPUT_PATH="/data/nas/yexin/workspace/shunian/model_training/FastVideo/data/50_hour_test_480p_49frames/videos2caption.json"
# 输出目录
OUTPUT_DIR="data/outputs/Hunyuan-Audio-Finetune-Hunyuan/result"
# 模型路径（如果为空，则使用异常数据中记录的检查点路径）
export MODEL_BASE=/data/nas/yexin/workspace/shunian/model
export MODEL_PATH=/data/nas/yexin/workspace/shunian/model_training/FastVideo/data/outputs/Hunyuan-Audio-Finetune-Hunyuan-audio-only/checkpoint-1000/diffusion_pytorch_model.safetensors

# 模型类型
MODEL_TYPE="hunyuan_audio"
# 推理步数
INFERENCE_STEPS=50
# 引导比例
GUIDANCE_SCALE=1.0
# 嵌入式引导比例
EMBEDDED_GUIDANCE_SCALE=6.0
# Flow shift参数
FLOW_SHIFT=17.0
# 是否反向Flow
FLOW_REVERSE="--flow_reverse"
# 序列并行大小
SP_SIZE=1

precision="bf16"

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 运行推理脚本
torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} --master_port ${MASTER_PORT} \
    fastvideo/sample/sample_at2v_hunyuan.py \
    --input_path ${INPUT_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --model_path ${MODEL_BASE} \
    --model_type ${MODEL_TYPE} \
    --dit-weight ${MODEL_PATH} \
    --num_inference_steps ${INFERENCE_STEPS} \
    --guidance_scale ${GUIDANCE_SCALE} \
    --embedded_guidance_scale ${EMBEDDED_GUIDANCE_SCALE} \
    --flow_shift ${FLOW_SHIFT} \
    ${FLOW_REVERSE} \
    --sp_size ${SP_SIZE} \
    --width 848 \
    --height 480 \
    --num_frames 49 \
    --fps 24 \
    --precision ${precision} \
    --data_dir ${DATA_DIR} \
    --use_v2c_format 

echo "推理完成，结果保存在 ${OUTPUT_DIR}" 