# 训练异常检测与分析工具

本工具集用于检测和分析训练过程中的异常情况（如loss和梯度范数的异常值）。整个工具集包含两个主要部分：

1. 训练时异常检测：嵌入在`train_audio.py`中的实时检测和记录功能
2. 离线分析工具：用于分析记录的异常数据的脚本`analyze_anomalies.py`

## 1. 训练时异常检测

在`train_audio.py`中，我们添加了自动检测和记录训练过程中异常的功能。当检测到异常时，相关数据会被保存到输出目录下的`anomalies`文件夹中。

### 异常检测参数

在`train_audio.py`的`main`函数中，我们初始化了异常检测器：

```python
anomaly_detector = AnomalyDetector(
    output_dir=args.output_dir,
    rank=rank,
    window_size=50,
    loss_threshold_factor=2.0,     # 降低阈值，更易检测波动
    grad_norm_threshold_factor=2.0, # 降低阈值，更易检测波动
    fixed_loss_threshold=0.3,      # 可选固定阈值
    fixed_grad_norm_threshold=1.0   # 可选固定阈值
)
```

可以根据需要调整以下参数：

- `window_size`: 滑动窗口大小，用于计算中位数和MAD（默认：50）
- `loss_threshold_factor`: loss阈值系数，用于动态阈值计算（中位数 + 系数 * MAD）
- `grad_norm_threshold_factor`: 梯度范数阈值系数
- `fixed_loss_threshold`: 固定loss阈值，如果设置则覆盖动态阈值
- `fixed_grad_norm_threshold`: 固定梯度范数阈值，如果设置则覆盖动态阈值

### 异常检测策略

我们使用多种策略来检测训练过程中的异常：

1. **绝对阈值检测**：如果设置了固定阈值（fixed_loss_threshold或fixed_grad_norm_threshold），则当loss或梯度范数超过这些阈值时被标记为异常。

2. **基于中位数的动态阈值检测**：使用滑动窗口中的中位数和MAD（中位数绝对偏差）计算动态阈值。与使用均值和标准差相比，中位数和MAD对异常值更加鲁棒，不易受极端值影响。
   - 动态阈值 = 中位数 + mad_factor * MAD
   - mad_factor = threshold_factor * 1.4826（正态分布下MAD与标准差的关系系数）

3. **相对变化率检测**：检测连续两个步骤之间的相对变化。当loss或梯度范数相对于前一步骤突然增加超过50%时，即使未超过上述阈值，也会被视为异常。

这种多策略检测方法可以捕获更多类型的异常模式，包括突然的波峰（spike）和持续上升的趋势，同时由于使用中位数和MAD，能够更好地抵抗数据中的噪声影响。

### 输出文件结构

异常检测器会在输出目录下创建以下文件结构：

```
output_dir/
  ├── anomalies/
  │    ├── anomalies_rank_0.jsonl   # 进程0检测到的异常记录
  │    ├── anomalies_rank_1.jsonl   # 进程1检测到的异常记录
  │    └── data/                    # 异常数据存储目录
  │         ├── step_XXX_rank_Y_TIMESTAMP/  # 每个异常的数据目录
  │         │    ├── latents.pt            # 潜在向量
  │         │    ├── encoder_hidden_states.pt  # 编码器隐藏状态
  │         │    └── ...                   # 其他张量数据
  │         └── ...
  └── ...
```

每个JSONL文件中的记录包含以下信息：

```json
{
  "step": 123,                  // 训练步骤
  "rank": 0,                    // 进程ID
  "timestamp": "2023-06-01T12:34:56",  // 时间戳
  "loss": 0.85,                 // 异常loss值
  "grad_norm": 2.9,             // 异常梯度范数值
  "batch_info": {               // 批次信息
    "latents_shape": [1, 32, 64, 64],  // 形状信息
    "latents_path": "anomalies/data/step_123_rank_0_20230601_123456/latents.pt"  // 文件路径
  },
  "model_checkpoint": "checkpoint-100",  // 最近的检查点
  "threshold_info": {           // 阈值信息
    "loss_median": 0.08,        // loss的中位数
    "loss_mad": 0.02,           // loss的MAD值
    "loss_threshold": 0.25,     // 计算得到的loss阈值
    "grad_norm_median": 0.25,   // 梯度范数的中位数
    "grad_norm_mad": 0.08,      // 梯度范数的MAD值
    "grad_norm_threshold": 0.6, // 计算得到的梯度范数阈值
    "loss_mean": 0.1,           // loss的均值（仅供参考）
    "loss_std": 0.05,           // loss的标准差（仅供参考）
    "grad_norm_mean": 0.3,      // 梯度范数的均值（仅供参考）
    "grad_norm_std": 0.1,       // 梯度范数的标准差（仅供参考）
    "relative_loss_change": 0.75,  // 相对前一步的loss变化率
    "relative_grad_change": 0.2    // 相对前一步的梯度范数变化率
  },
  "data_dir": "anomalies/data/step_123_rank_0_20230601_123456"  // 数据目录相对路径
}
```

## 2. 离线分析工具

`analyze_anomalies.py`脚本用于分析记录的异常数据。它会加载异常记录，重新执行前向和反向传播，并收集详细的梯度信息。

### 使用方法

```bash
python scripts/analysis/analyze_anomalies.py \
  --anomaly_dir path/to/output_dir/anomalies \
  --output_dir path/to/analysis_results \
  --model_path path/to/model \
  --model_type hunyuan_audio \
  --device cuda \
  --visualize
```

参数说明：

- `--anomaly_dir`: 异常数据目录（包含anomalies_rank_*.jsonl文件的目录）
- `--output_dir`: 分析结果输出目录
- `--model_path`: 模型路径
- `--model_type`: 模型类型，默认为"hunyuan_audio"
- `--device`: 运行设备，默认为"cuda"
- `--visualize`: 是否生成可视化图表
- `--analyze_specific_step`: 指定分析某个特定步骤的异常（可选）

### 分析输出

分析工具会生成以下输出：

1. 每个异常的详细分析结果（JSON文件）
   - 原始loss和分析时的loss
   - 梯度信息，包括按范数排序的top梯度
   - 激活值信息（如果启用）

2. 汇总分析（analysis_summary.json）
   - 异常数量
   - 平均loss
   - 平均梯度范数

3. 可视化结果（如果启用`--visualize`）
   - 异常概览图（loss和梯度范数）
   - 每个异常的Top-10梯度分布

## 分析异常原因的建议

1. 比较训练时和分析时的loss差异
2. 检查哪些模块的梯度范数异常高
3. 观察这些模块的参数和激活值
4. 检查输入数据是否有特殊特征
5. 分析异常检测的阈值信息，查看中位数和MAD值，了解异常是由绝对值过高还是相对变化率过大引起的
6. 比较中位数和均值的差异，如果差异很大，可能表明数据中存在特别的异常值
7. 如果可能，重试相同批次的训练，但使用不同的随机种子

## 为什么使用中位数和MAD代替均值和标准差？

均值和标准差容易受到极端值（outliers）的影响。在检测异常的场景中，之前的异常值可能会扭曲均值和标准差的计算，导致后续的异常检测变得不准确。

中位数和MAD（中位数绝对偏差）是更加鲁棒的统计量：
- 中位数代表数据的中心位置，不受极端值影响
- MAD衡量数据的离散程度，计算时先减去中位数再取绝对值，最后计算这些绝对值的中位数
- 在正态分布下，MAD ≈ 0.6745 × 标准差，因此我们使用了调整因子1.4826(≈1/0.6745)

这种方法使得异常检测系统更加鲁棒，能够更准确地识别真正的异常，而不会被之前检测到的异常所干扰。

## 注意事项

1. 实时异常检测会在每个训练步骤后检查loss和梯度范数
2. 为了节省磁盘空间，仅保存张量的形状信息和文件路径，而不是完整的张量
3. 多进程训练中，每个进程独立检测和记录异常
4. 分析工具使用的计算环境应与训练环境一致，以确保结果可比
5. 降低检测阈值将导致更多的数据被标记为异常，可能会增加存储开销 