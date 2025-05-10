import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob # Import glob for file searching
import warnings # Import warnings to handle the torch.load warning
import numpy as np # 导入 numpy 用于计算 min/max
import matplotlib.colors as mcolors # 导入 colors 用于 Normalize
import matplotlib.cm as cm # 导入 cm 用于 ScalarMappable
from tqdm import tqdm

# Suppress the specific FutureWarning from torch.load
warnings.filterwarnings("ignore", category=FutureWarning, module='torch.serialization')

# Define the directory containing the loss files
# loss_dir = "/data/nas/yexin/workspace/shunian/model_training/FastVideo/data/outputs/Hunyuan-Audio-Finetune-Hunyuan-ai2v-49frames-200_hour_480p_49frames_eng-0503-debug-yexin3/sample_losses"
loss_dir = "/data/nas/yexin/workspace/shunian/model_training/FastVideo/data/outputs/Hunyuan-Audio-Finetune-Hunyuan-ai2v-49frames-200_hour_480p_49frames_eng-0510-facemask/sample_losses"

# Define the output directory for the plot
output_dir = "model_training/FastVideo/tests_facemask/visualizations"
# Define the output filename for the average heatmap
average_output_filename = os.path.join(output_dir, "average_heatmap_no_nan.png") # Updated filename

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Find all heatmap.pt files in the directory
heatmap_files = glob.glob(os.path.join(loss_dir, "*heatmap.pt"))

heatmap_files = [file for file in heatmap_files if "weighted" in file]
# heatmap_files = [file for file in heatmap_files if "weighted" not in file]

if not heatmap_files:
    print(f"No '*heatmap.pt' files found in {loss_dir}")
    exit()

print(f"Found {len(heatmap_files)} heatmap files. Processing...")

all_tensors = []
first_tensor_shape = None
nan_count = 0
skipped_count = 0

# Load all tensors, skipping those with NaN
for loss_file_path in tqdm(heatmap_files, desc="Processing heatmaps"):
    try:
        # Consider adding weights_only=True for security if applicable
        loss_data = torch.load(loss_file_path, map_location='cpu', weights_only=False) # Explicitly set weights_only for now

        # Check if the loaded data is a tensor or a dict containing the tensor
        if not isinstance(loss_data, torch.Tensor):
            if isinstance(loss_data, dict) and 'heatmap' in loss_data:
                 loss_data = loss_data['heatmap']
            else:
                print(f"Warning: Skipping {loss_file_path}. Loaded data is not a tensor or a dict with a 'heatmap' key. Found type: {type(loss_data)}")
                skipped_count += 1
                continue # Skip this file

        # Ensure it's a 2D tensor
        if loss_data.ndim != 2:
            print(f"Warning: Skipping {loss_file_path}. Tensor is not 2D, has dimensions {loss_data.shape}.")
            skipped_count += 1
            continue # Skip this file

        # Check for NaN values
        if torch.isnan(loss_data).any():
            print(f"Warning: Skipping {loss_file_path}. Tensor contains NaN values.")
            nan_count += 1
            continue # Skip this file

        # Check if shapes are consistent
        if first_tensor_shape is None:
            first_tensor_shape = loss_data.shape
        elif loss_data.shape != first_tensor_shape:
            print(f"Warning: Skipping {loss_file_path}. Tensor shape {loss_data.shape} does not match first tensor shape {first_tensor_shape}.")
            skipped_count += 1
            continue # Skip this file

        all_tensors.append(loss_data)

    except FileNotFoundError:
        print(f"Warning: File not found at {loss_file_path}. Skipping.")
        skipped_count += 1
    except Exception as e:
        print(f"An error occurred while processing {loss_file_path}: {e}. Skipping.")
        skipped_count += 1

print(f"Finished processing. Skipped {nan_count} files due to NaN values, {skipped_count} due to other errors/inconsistencies.")

if not all_tensors:
    print("No valid heatmap tensors (without NaN and with consistent shape) found or loaded. Exiting.")
    exit()

print(f"Calculating average heatmap from {len(all_tensors)} valid tensors...")


# selected_indices = [1, 6, 42, 137] # 使用更清晰的变量名
# 使用列表推导式根据索引选择张量
# all_tensors = [all_tensors[i] for i in selected_indices]

# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import math

# # 确定网格布局
# num_plots = len(all_tensors)
# ncols = 2 # 每行显示2个图
# nrows = math.ceil(num_plots / ncols)

# # 创建子图
# fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5)) # 调整 figsize 以获得更好的视觉效果
# # 如果只有一个图，axes 不是数组，将其转换为数组
# if num_plots == 1:
#     axes = [axes]
# else:
#     axes = axes.flatten() # 将 axes 数组展平成一维，方便迭代

# # 绘制每个选定的 heatmap
# for i, tensor in enumerate(all_tensors):
#     ax = axes[i]
#     # 确保 tensor 是 numpy array 供 seaborn 使用
#     heatmap_data = tensor.cpu().numpy() if hasattr(tensor, 'cpu') else tensor.numpy()
#     # 找到所有张量中的全局 min 和 max
#     all_heatmap_data = [t.cpu().numpy() if hasattr(t, 'cpu') else t.numpy() for t in all_tensors]
#     global_vmin = np.min([data.min() for data in all_heatmap_data])
#     global_vmax = np.max([data.max() for data in all_heatmap_data])

#     sns.heatmap(heatmap_data, cmap='viridis', ax=ax, cbar=False, vmin=global_vmin, vmax=global_vmax) # 使用全局 vmin/vmax 并禁用单独的 cbar
#     # original_index = selected[i] # 获取原始索引
#     # ax.set_title(f"Heatmap index {original_index}")
#     ax.axis('off') # 关闭坐标轴

# # 如果子图数量多于实际图像数量，隐藏多余的子图
# for j in range(num_plots, nrows * ncols):
#     axes[j].axis('off')

# # 调整布局为 colorbar 留出空间
# plt.tight_layout(rect=[0.03, 0.03, 0.92, 0.95]) # rect=[left, bottom, right, top] - 调整为右侧留空间

# # 创建 ScalarMappable 用于共享 colorbar
# norm = mcolors.Normalize(vmin=global_vmin, vmax=global_vmax)
# sm = cm.ScalarMappable(cmap='viridis', norm=norm)
# sm.set_array([]) # 需要设置一个空数组

# # 在图的右侧添加 colorbar 轴
# # fig.add_axes([left, bottom, width, height]) in figure coordinates
# cbar_ax = fig.add_axes([0.93, 0.15, 0.03, 0.7]) # 调整 left 值将轴移到右侧
# fig.colorbar(sm, cax=cbar_ax)
# cbar_ax.yaxis.set_ticks_position('right') # 将刻度放在右侧

# # 保存为 PDF 文件
# output_pdf_filename = os.path.join(output_dir, "selected_heatmaps_shared_cbar_right.pdf") # 更新文件名
# plt.savefig(output_pdf_filename, format='pdf')
# print(f"Selected heatmaps with shared colorbar on the right saved as PDF to {output_pdf_filename}")
# plt.close(fig) # 关闭图形，释放内存

# # 删除后续的单个图像保存循环，因为它已被上面的代码取代

for i, tensor in enumerate(all_tensors):
    plt.figure(figsize=(10, 8))
    sns.heatmap(tensor, cmap='viridis') # You can choose other color maps
    # plt.title(f'Average Heatmap ({len(all_tensors)} files, NaN excluded)')
    # plt.xlabel("Dimension 2")
    # plt.ylabel("Dimension 1")

    # disable the axis
    plt.axis('off')

    # Save the plot
    output_filename = os.path.join(output_dir, f"weighted_heatmap_{i}.png")
    plt.savefig(output_filename)
    print(f"Heatmap {i} saved to {output_filename}")

# Calculate the average tensor
average_tensor = torch.mean(torch.stack(all_tensors), dim=0)

# Convert average tensor to numpy array
average_loss_numpy = average_tensor.numpy()

# Create the heatmap for the average
try:
    plt.figure(figsize=(10, 8))
    sns.heatmap(average_loss_numpy, cmap='viridis') # You can choose other color maps
    plt.title(f'Average Heatmap ({len(all_tensors)} files, NaN excluded)')
    plt.xlabel("Dimension 2")
    plt.ylabel("Dimension 1")

    # Save the plot
    plt.savefig(average_output_filename)
    print(f"Average heatmap saved to {average_output_filename}")

    # # Optionally, display the plot
    # plt.show()

except Exception as e:
    print(f"An error occurred during plotting or saving the average heatmap: {e}")
