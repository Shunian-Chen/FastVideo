# 异常推理工具

这个工具用于对训练过程中检测到的异常样本进行推理，生成可视化结果，帮助分析和理解异常原因。

## 背景

在训练过程中，`train_audio.py`会自动检测和记录训练过程中的异常情况，例如：

1. Loss值异常增大
2. 梯度范数异常增大
3. Loss或梯度范数突然激增

当检测到异常时，训练脚本会保存相关数据（包括模型输入、音频嵌入等）到输出目录下的`anomalies`文件夹中。本工具可以对这些异常样本进行推理，生成视频结果，帮助分析异常原因。

## 脚本功能

本工具包含两个脚本：

1. `inference_anomaly.py` - 主要推理脚本，可以加载异常数据并进行推理
2. `run_inference_anomaly.sh` - 运行脚本，用于简化参数设置

## 如何使用

### 1. 使用运行脚本（推荐）

运行脚本提供了一组默认参数，可以直接运行：

```bash
# 使用1个GPU运行
bash scripts/analysis/run_inference_anomaly.sh 1

# 使用4个GPU运行，并使用特定端口
bash scripts/analysis/run_inference_anomaly.sh 4 29506
```

你可以根据需要修改`run_inference_anomaly.sh`中的参数：

```bash
# 异常数据目录
ANOMALY_DIR="outputs/anomalies"
# 输出目录
OUTPUT_DIR="outputs/anomaly_inference"
# 模型路径（如果为空，则使用异常数据中记录的检查点路径）
MODEL_PATH=""
# 模型类型
MODEL_TYPE="hunyuan_audio"
# 推理步数
INFERENCE_STEPS=50
# ...等等
```

### 2. 直接使用Python脚本

如果需要更灵活的参数设置，可以直接使用Python脚本：

```bash
torchrun --nnodes=1 --nproc_per_node=1 scripts/analysis/inference_anomaly.py \
    --anomaly_dir outputs/anomalies \
    --output_dir outputs/anomaly_inference \
    --model_type hunyuan_audio \
    --inference_steps 50 \
    --guidance_scale 1.0 \
    --embedded_guidance_scale 6.0 \
    --flow_shift 17.0 \
    --flow_reverse \
    --width 1280 \
    --height 720 \
    --num_frames 125
```

## 参数说明

### 主要参数

- `anomaly_dir` - 异常数据目录，通常是训练输出目录下的`anomalies`文件夹
- `output_dir` - 输出目录，用于保存推理结果
- `model_path` - 模型路径，如果不指定则使用异常数据中记录的检查点路径
- `model_type` - 模型类型，支持`hunyuan_audio`、`hunyuan`、`hunyuan_hf`和`mochi`
- `anomaly_id` - 指定异常ID（如果指定，则只处理该ID的异常）

### 推理参数

- `inference_steps` - 推理步数，默认为50
- `guidance_scale` - 引导比例，默认为6.0
- `embedded_guidance_scale` - 嵌入式引导比例
- `flow_shift` - Flow shift参数，默认为5.0
- `flow_reverse` - 是否反向Flow
- `width` - 视频宽度，默认为1280
- `height` - 视频高度，默认为720
- `num_frames` - 视频帧数，默认为125
- `fps` - 视频帧率，默认为24

## 输出结果

推理完成后，结果会保存在`output_dir`目录下，包括：

- `anomaly_step_{step}_{timestamp}.mp4` - 推理生成的视频文件
- 如果视频保存失败，会将帧保存到`anomaly_step_{step}_{timestamp}_frames`目录下

## 注意事项

1. 推理过程可能需要较高的GPU内存，如果遇到OOM问题，可以尝试降低视频分辨率或帧数
2. 如果没有指定`model_path`，脚本会使用异常数据中记录的检查点路径，确保检查点文件存在
3. 对于不同的模型类型，可能需要调整推理参数以获得最佳结果 