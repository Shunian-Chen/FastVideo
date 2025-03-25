import torch
import torch.nn as nn
import torch.nn.init as init
import math
# 定义一个简单的线性层
linear_layer = nn.Linear(in_features=10, out_features=5)

# 手动初始化，复制默认行为  (对于Linear层)
linear_layer_manual = nn.Linear(in_features=10, out_features=5)
init.kaiming_uniform_(linear_layer_manual.weight, a=math.sqrt(5)) # fan_in, preserve_rng=False
if linear_layer_manual.bias is not None:
    fan_in, _ = init._calculate_fan_in_and_fan_out(linear_layer_manual.weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    init.uniform_(linear_layer_manual.bias, -bound, bound)


# 比较两种初始化方式的结果
print("Weights are equal:", torch.allclose(linear_layer.weight.data, linear_layer_manual.weight.data))
print("Bias are equal:", torch.allclose(linear_layer.bias.data, linear_layer_manual.bias.data))

