#!/bin/bash

# 设置GPU数量，默认为1
export CUDA_VISIBLE_DEVICES=5
# 设置GPU数量，根据CUDA_VISIBLE_DEVICES设置
NUM_GPUS=$(echo ${CUDA_VISIBLE_DEVICES} | tr ',' '\n' | wc -l)
# 设置主端口
MASTER_PORT=29510

echo "NUM_GPUS: ${NUM_GPUS}"
echo "MASTER_PORT: ${MASTER_PORT}"

# 数据目录结构
# - videos2caption.json: 包含视频标题与音频嵌入文件路径的JSON文件
# - audio_emb/: 存放音频嵌入文件 (.pt)
# - audios/: 存放对应的音频文件 (.wav)
# 数据目录
DATA_DIR="/data/nas/yexin/workspace/shunian/model_training/FastVideo/data/evaluation_480p_49frames"
INPUT_PATH="/data/nas/yexin/workspace/shunian/model_training/FastVideo/data/evaluation_480p_49frames/videos2caption_with_first_frame.json"

# 模型基础路径
export MODEL_BASE=/data/nas/yexin/workspace/shunian/model
# 模型输出的基础目录 (包含所有 checkpoints)
MODEL_OUTPUT_BASE_DIR="/data/nas/yexin/workspace/shunian/model/hunyuan-video-i2v-720p/transformers/mp_rank_00_model_states.pt"
# 从 MODEL_OUTPUT_BASE_DIR 提取模型名称
MODEL_NAME="hunyuan-video-i2v-720p-stability-audio-disabled-test"

# 要测试的 Checkpoint 列表

# 每个 Checkpoint 测试的视频数量
MAX_SAMPLES=10

# 模型类型
MODEL_TYPE="hunyuan_audio_i2v"
# 推理步数
INFERENCE_STEPS=50
# 引导比例
GUIDANCE_SCALE=1.0
# 嵌入式引导比例
EMBEDDED_GUIDANCE_SCALE=6.0
# Flow shift参数
FLOW_SHIFT=7.0
# 是否反向Flow
FLOW_REVERSE="--flow_reverse"
# 序列并行大小
SP_SIZE=1

precision="bf16"
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


# 循环遍历每个 Checkpoint

# 创建日志目录
LOG_DIR="logs/inference_logs/${MODEL_NAME}"
mkdir -p ${LOG_DIR}

# 设置日志路径 (仍然包含样本数以区分日志文件)
LOG_PATH="${LOG_DIR}/${MODEL_NAME}_run_with_${MAX_SAMPLES}_samples.log"

# 输出目录 (不再包含样本数后缀，确保一致性)
OUTPUT_DIR="data/outputs/result/${MODEL_NAME}/${MODEL_NAME}_evaluation"
mkdir -p ${OUTPUT_DIR}

# 运行推理脚本
echo "运行命令: torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} --master_port ${MASTER_PORT} fastvideo/sample/sample_ait2v_hunyuan.py ..."
python -m torch.distributed.run --nnodes=1 --nproc_per_node=${NUM_GPUS} --master_port ${MASTER_PORT} \
    fastvideo/sample/sample_ait2v_hunyuan.py \
    --input_path ${INPUT_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --model_path ${MODEL_BASE} \
    --model_type ${MODEL_TYPE} \
    --dit-weight ${MODEL_OUTPUT_BASE_DIR} \
    --num_inference_steps ${INFERENCE_STEPS} \
    --guidance_scale ${GUIDANCE_SCALE} \
    --embedded_guidance_scale ${EMBEDDED_GUIDANCE_SCALE} \
    --flow_shift ${FLOW_SHIFT} \
    ${FLOW_REVERSE} \
    --sp_size ${SP_SIZE} \
    --width 864 \
    --height 480 \
    --num_frames 49 \
    --fps 24 \
    --precision ${precision} \
    --data_dir ${DATA_DIR} \
    --use_v2c_format \
    --max_samples ${MAX_SAMPLES} \
    --i2v_mode \
    --i2v_resolution 540p \
    --i2v_stability \
    # > ${LOG_PATH} 2>&1

echo "Checkpoint ${CHECKPOINT_NAME} (限制 ${MAX_SAMPLES} samples) 推理完成，结果保存在 ${OUTPUT_DIR}"
echo "日志文件: ${LOG_PATH}"
echo "======================================================"
echo ""


echo "所有 Checkpoint 测试完成。" 