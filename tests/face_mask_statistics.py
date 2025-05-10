import torch
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
import sys
import argparse
from tqdm import tqdm
import multiprocessing
from functools import partial

# --- 配置 ---
# 这些路径可以通过命令行参数覆盖
DEFAULT_JSON_PATH = "/data/nas/yexin/workspace/shunian/model_training/FastVideo/data/200_hour_480p_49frames_eng/videos2caption_cleaned.json"
DEFAULT_OUTPUT_DIR = "/data/nas/yexin/workspace/shunian/model_training/FastVideo/tests/mask_statistics_output" # 输出统计结果和图表的目录
DEFAULT_NUM_WORKERS = 16 # 默认使用所有 CPU 核心

# --- 常量 (来自 latent_datasets_audio_i2v.py) ---
BACKGROUND_VALUE = 0.1
FACE_MASK_VALUE = 0.5
LIP_MASK_VALUE = 1.0

# --- 日志配置 ---
logger.remove()
logger.add(sys.stdout, level="INFO")

# --- 辅助函数 ---
def get_samples_around_percentiles(data, key, percentiles, num_samples=10):
    """
    查找并返回指定百分位点周围的样本。

    Args:
        data (list): 包含样本信息的字典列表。
        key (str): 用于排序和计算百分位的字典键 (例如 'face_area_percent')。
        percentiles (list): 需要查找的百分位列表 (例如 [1, 20, 40, 60, 80, 99])。
        num_samples (int): 每个百分位点附近要获取的样本数量。

    Returns:
        dict: 键是百分位，值是包含样本信息（含排名）的列表。
    """
    if not data:
        logger.warning(f"输入数据为空，无法计算 {key} 的百分位样本。")
        return {p: [] for p in percentiles}

    # 确保数据按指定键排序
    try:
        sorted_data = sorted(data, key=lambda x: x[key])
    except KeyError:
        logger.error(f"提供的键 '{key}' 不存在于样本数据中。")
        return {p: [] for p in percentiles}
    except Exception as e:
        logger.error(f"根据键 '{key}' 排序数据时出错: {e}")
        return {p: [] for p in percentiles}

    n = len(sorted_data)
    if n == 0:
        logger.warning(f"排序后数据为空，无法计算 {key} 的百分位样本。")
        return {p: [] for p in percentiles}

    results = {}

    for p in percentiles:
        if not (0 <= p <= 100):
            logger.warning(f"百分位 {p} 无效，应在 0 到 100 之间。跳过。")
            results[p] = []
            continue

        # 计算目标索引 (0-based)。使用 round 确保更接近预期位置，尤其是在数据量少时。
        # (n-1) 是因为索引是从0开始的。
        target_idx = int(round(p / 100.0 * (n - 1)))

        # 计算实际获取样本的切片边界
        # 目标是获取 num_samples 个样本，以 target_idx 为中心（或附近）
        start_idx = max(0, target_idx - (num_samples // 2))
        end_idx = start_idx + num_samples

        # 调整边界，确保不会超出列表范围
        if end_idx > n:
            end_idx = n
            start_idx = max(0, n - num_samples) # 如果末尾不足 num_samples，则从倒数第 num_samples 个开始取

        percentile_samples = []
        for i in range(start_idx, end_idx):
            # 添加排名 (1-based) 到样本信息中以便参考
            sample_info = sorted_data[i].copy()
            sample_info['rank'] = i + 1 # 排名从 1 开始
            percentile_samples.append(sample_info)
        results[p] = percentile_samples

    return results

# --- 工作函数 (用于多进程) ---
def process_mask_file(task_args):
    """处理单个 mask 文件，计算面积占比。"""
    mask_coord_path, sample_identifier = task_args

    if not os.path.exists(mask_coord_path):
        # logger.error(f"Mask 坐标文件未找到: {mask_coord_path}。跳过此样本。") # 不在工作进程中打日志
        return {"id": sample_identifier, "status": "skipped_no_file"}

    try:
        mask_data = torch.load(mask_coord_path, map_location="cpu")

        if isinstance(mask_data, dict) and "original_height" in mask_data and "frames" in mask_data:
            original_h = mask_data["original_height"]
            original_w = mask_data["original_width"]
            frame_coords_list = mask_data["frames"]
            num_frames_in_file = len(frame_coords_list)

            if original_h <= 0 or original_w <= 0:
                 # logger.warning(f"样本 {sample_identifier} 原始尺寸无效 ({original_h}x{original_w})，跳过。")
                 return {"id": sample_identifier, "status": "error_invalid_dim"}

            frame_total_area = float(original_h * original_w)
            if frame_total_area == 0:
                 # logger.warning(f"样本 {sample_identifier} 帧总面积为零，无法计算占比，跳过。")
                 return {"id": sample_identifier, "status": "error_zero_area"}

            total_sample_face_percent = 0.0
            total_sample_lip_percent = 0.0
            valid_frames_count = 0

            for t_idx in range(num_frames_in_file):
                if t_idx >= 0 and t_idx < len(frame_coords_list) and frame_coords_list[t_idx] is not None:
                    coords = frame_coords_list[t_idx]
                    if len(coords) == 8:
                        fx1, fy1, fx2, fy2, lx1, ly1, lx2, ly2 = [int(c) for c in coords]
                        fx1, fx2 = max(0, fx1), min(original_w, fx2)
                        fy1, fy2 = max(0, fy1), min(original_h, fy2)
                        lx1, lx2 = max(0, lx1), min(original_w, lx2)
                        ly1, ly2 = max(0, ly1), min(original_h, ly2)

                        current_frame_face_area_pixels = 0
                        current_frame_lip_area_pixels = 0

                        is_face_valid = fx2 > fx1 and fy2 > fy1
                        is_lip_valid = lx2 > lx1 and ly2 > ly1

                        if is_lip_valid:
                            current_frame_lip_area_pixels = (lx2 - lx1) * (ly2 - ly1)

                        if is_face_valid:
                            total_face_rect_area_pixels = (fx2 - fx1) * (fy2 - fy1)
                            if is_lip_valid:
                                current_frame_face_area_pixels = total_face_rect_area_pixels - current_frame_lip_area_pixels
                                current_frame_face_area_pixels = max(0, current_frame_face_area_pixels)
                            else:
                                current_frame_face_area_pixels = total_face_rect_area_pixels

                        current_frame_face_percent = current_frame_face_area_pixels / frame_total_area
                        current_frame_lip_percent = current_frame_lip_area_pixels / frame_total_area

                        total_sample_face_percent += current_frame_face_percent
                        total_sample_lip_percent += current_frame_lip_percent
                        valid_frames_count += 1

            if valid_frames_count > 0:
                avg_face_percent = total_sample_face_percent / valid_frames_count
                avg_lip_percent = total_sample_lip_percent / valid_frames_count
                return {"id": sample_identifier, "face_area_percent": avg_face_percent, "lip_area_percent": avg_lip_percent, "valid": True, "status": "success"}
            else:
                # logger.warning(f"样本 {sample_identifier} 没有有效的 mask 帧，无法计算平均占比。")
                return {"id": sample_identifier, "face_area_percent": 0.0, "lip_area_percent": 0.0, "valid": False, "status": "error_no_valid_frames"}
        else:
            # logger.error(f"Mask 文件 {mask_coord_path} 不是预期的字典格式。跳过此样本。")
            return {"id": sample_identifier, "face_area_percent": 0.0, "lip_area_percent": 0.0, "valid": False, "status": "error_bad_format"}
    except Exception as e:
        # logger.error(f"处理 mask 文件 {mask_coord_path} 时出错: {e}", exc_info=False)
        # 可以在这里返回更详细的错误信息，但为了简化，只标记为错误
        return {"id": sample_identifier, "status": "error_exception", "error_msg": str(e)}

def calculate_mask_statistics(json_path, face_mask_coord_dir, output_dir, num_workers):
    """使用多进程加载数据，计算 face 和 lip 区域面积占比分布"""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录: {os.path.abspath(output_dir)}")
    logger.info(f"使用 JSON 文件: {json_path}")
    logger.info(f"Mask 坐标目录: {face_mask_coord_dir}")
    logger.info(f"使用 {num_workers} 个工作进程。")

    try:
        with open(json_path, "r") as f:
            data_anno = json.load(f)
        logger.info(f"成功加载 JSON 文件，包含 {len(data_anno)} 条记录")
    except FileNotFoundError:
        logger.error(f"JSON 文件未找到: {json_path}")
        return
    except json.JSONDecodeError:
        logger.error(f"无法解析 JSON 文件: {json_path}")
        return

    # --- 构建任务列表 ---
    tasks = []
    skipped_json_entries = 0
    logger.info("开始构建处理任务列表...")
    for i, data_item in enumerate(data_anno):
        face_mask_coord_file = data_item.get("face_emb_path")
        if not face_mask_coord_file:
            skipped_json_entries += 1
            continue

        # 修正路径
        if face_mask_coord_file.endswith(".png"):
             face_mask_coord_file = face_mask_coord_file.replace(".png", ".pt")
        elif not face_mask_coord_file.endswith(".pt"):
             potential_pt_file = face_mask_coord_file + ".pt"
             potential_pt_path = os.path.join(face_mask_coord_dir, potential_pt_file)
             original_path = os.path.join(face_mask_coord_dir, face_mask_coord_file)
             if os.path.exists(potential_pt_path):
                 face_mask_coord_file = potential_pt_file
             # 如果修正后的.pt不存在，工作函数会在尝试打开时处理

        mask_coord_path = os.path.join(face_mask_coord_dir, face_mask_coord_file)
        sample_identifier = os.path.splitext(face_mask_coord_file)[0]
        tasks.append((mask_coord_path, sample_identifier))

    logger.info(f"任务列表构建完成，共 {len(tasks)} 个任务。跳过了 {skipped_json_entries} 个缺少 mask 路径的 JSON 条目。")

    # --- 多进程处理 ---
    sample_data = []
    processed_count = 0
    error_count = 0
    skipped_count = skipped_json_entries # 从 JSON 解析时跳过的

    logger.info("开始多进程处理 Mask 文件...")
    with multiprocessing.Pool(processes=num_workers) as pool:
        # 使用 imap_unordered 以获得更好的进度反馈和潜在的性能提升
        results_iterator = pool.imap_unordered(process_mask_file, tasks)
        # 使用 tqdm 显示进度条
        for result in tqdm(results_iterator, total=len(tasks), desc="处理 Mask 文件"):
            processed_count += 1 # 无论成功失败，都算处理了一个任务
            if result is not None:
                if result.get("valid", False):
                    sample_data.append(result)
                else:
                    # 根据 status 区分错误和跳过
                    if "error" in result.get("status", ""):
                        error_count += 1
                        # 可以选择性地记录一些错误
                        # if result.get("status") == "error_exception":
                        #     logger.warning(f"处理 {result.get('id', '未知')} 时出错: {result.get('error_msg', '未知错误')}")
                    elif "skipped" in result.get("status", ""):
                         skipped_count += 1 # 文件不存在等情况
                    else: # 其他非 valid 情况视为错误
                         error_count += 1
            else:
                # 理论上不应该发生，除非 worker 内部完全崩溃且未返回
                error_count += 1
                logger.error("工作进程返回了 None，可能存在严重错误。")

    logger.info(f"多进程处理完成。总任务数: {len(tasks)}, 有效结果: {len(sample_data)}, 错误数: {error_count}, 跳过数(含文件不存在): {skipped_count}")

    if not sample_data:
        logger.error("没有处理成功任何有效样本，无法进行统计分析。")
        return

    logger.info(f"有效样本数量进行统计: {len(sample_data)}")

    # --- 后续统计分析保持不变，使用 sample_data ---
    valid_face_percents = np.array([s["face_area_percent"] for s in sample_data]) # 注意这里是 sample_data
    valid_lip_percents = np.array([s["lip_area_percent"] for s in sample_data])

    # --- 可视化 (百分比) ---
    logger.info("开始生成面积占比分布直方图...")
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    face_hist_data = valid_face_percents * 100
    if len(face_hist_data) > 0: # 检查是否有数据
        face_hist, face_bins, _ = plt.hist(face_hist_data, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title(f'Face Area Distribution (N={len(valid_face_percents)})')
        plt.xlabel('Average Area Percentage (%)')
        plt.ylabel('Sample Count')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        face_mean_percent = np.mean(face_hist_data)
        face_median_percent = np.median(face_hist_data)
        plt.axvline(face_mean_percent, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {face_mean_percent:.2f}%')
        plt.axvline(face_median_percent, color='g', linestyle='dashed', linewidth=1, label=f'Median: {face_median_percent:.2f}%')
        plt.legend()
    else:
        plt.title('Face Area Distribution')
        plt.text(0.5, 0.5, 'No valid face area data', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

    plt.subplot(1, 2, 2)
    non_zero_lip_percents = valid_lip_percents[valid_lip_percents > 1e-9]
    zero_lip_count = len(valid_lip_percents) - len(non_zero_lip_percents)
    if len(non_zero_lip_percents) > 0:
        lip_hist_data = non_zero_lip_percents * 100
        lip_hist, lip_bins, _ = plt.hist(lip_hist_data, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
        plt.title(f'Lip Area Distribution (Area > 0, N={len(non_zero_lip_percents)})')
        plt.xlabel('Average Area Percentage (%)')
        plt.ylabel('Sample Count')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        lip_mean_percent = np.mean(lip_hist_data)
        lip_median_percent = np.median(lip_hist_data)
        plt.axvline(lip_mean_percent, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {lip_mean_percent:.2f}%')
        plt.axvline(lip_median_percent, color='g', linestyle='dashed', linewidth=1, label=f'Median: {lip_median_percent:.2f}%')
        plt.legend()
        plt.text(0.95, 0.95, f'There are {zero_lip_count} samples with lip area approximately equal to 0',
                 horizontalalignment='right', verticalalignment='top',
                 transform=plt.gca().transAxes, fontsize=9, color='gray')
    else:
        plt.title('Lip Area Distribution')
        plt.text(0.5, 0.5, f'No samples with lip area > 0 (There are {zero_lip_count} samples with lip area approximately equal to 0)',
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Face and Lip Mask Average Area Percentage Distribution Statistics", fontsize=16)
    hist_path = os.path.join(output_dir, "mask_area_percentage_distribution.png")
    try:
        plt.savefig(hist_path)
        logger.info(f"面积占比分布直方图已保存至: {hist_path}")
    except Exception as e:
        logger.error(f"保存直方图失败: {e}")
    plt.close()

    # --- 记录极值和百分位样本 (百分比) ---
    logger.info("开始记录面积占比极值和百分位样本...")
    if not sample_data: # 如果没有有效数据，则跳过排序和记录
        logger.warning("无有效样本数据，无法记录极值或百分位样本。")
        return

    # --- 人脸面积统计 ---
    # 按人脸面积占比排序 (升序)
    face_sorted_samples = sorted(sample_data, key=lambda x: x["face_area_percent"])
    n_face = len(face_sorted_samples)

    logger.info("--- 人脸面积占比最小的 10 个有效样本 ---")
    for i, s in enumerate(face_sorted_samples[:10]): # 只取前10个
        logger.info(f"Rank {i+1}/{n_face}. ID: {s['id']}, 平均 Face 占比: {s['face_area_percent']:.4%}, 平均 Lip 占比: {s['lip_area_percent']:.4%}")

    logger.info("--- 人脸面积占比最大的 10 个有效样本 ---")
    for i, s in enumerate(reversed(face_sorted_samples[-10:])): # 只取后10个
        rank = n_face - i
        logger.info(f"Rank {rank}/{n_face}. ID: {s['id']}, 平均 Face 占比: {s['face_area_percent']:.4%}, 平均 Lip 占比: {s['lip_area_percent']:.4%}")

    # 计算并记录人脸面积特定百分位的样本
    face_percentiles_to_log = [1, 20, 40, 60, 80, 99] # 选择的百分位点
    face_percentile_samples = get_samples_around_percentiles(face_sorted_samples, 'face_area_percent', face_percentiles_to_log, num_samples=10)
    for p, samples in face_percentile_samples.items():
        logger.info(f"--- 人脸面积占比约 {p}% 百分位点的 10 个样本 (共 {n_face} 个有效样本) ---")
        if samples:
            target_rank = int(round(p / 100.0 * (n_face - 1))) + 1
            logger.info(f"    (目标排名约: {target_rank}/{n_face})")
            for s in samples:
                logger.info(f"Rank {s['rank']}/{n_face}. ID: {s['id']}, 平均 Face 占比: {s['face_area_percent']:.4%}, 平均 Lip 占比: {s['lip_area_percent']:.4%}")
        else:
            logger.warning(f"    未能获取 {p}% 百分位点的样本。")

    # --- 嘴唇面积统计 ---
    # 按嘴唇面积占比排序 (升序)
    lip_sorted_samples = sorted(sample_data, key=lambda x: x["lip_area_percent"])
    n_lip = len(lip_sorted_samples)

    logger.info("--- 嘴唇面积占比最小的 10 个有效样本 (可能为0) ---")
    for i, s in enumerate(lip_sorted_samples[:10]): # 只取前10个
         logger.info(f"Rank {i+1}/{n_lip}. ID: {s['id']}, 平均 Face 占比: {s['face_area_percent']:.4%}, 平均 Lip 占比: {s['lip_area_percent']:.4%}")

    # 嘴唇面积占比最大的样本 (只考虑 > 0 的)
    non_zero_lip_samples = sorted([s for s in sample_data if s['lip_area_percent'] > 1e-9], key=lambda x: x["lip_area_percent"])
    n_lip_gt_zero = len(non_zero_lip_samples)

    if non_zero_lip_samples:
         logger.info(f"--- 嘴唇面积占比最大的 10 个有效样本 (占比 > 0, 共 {n_lip_gt_zero} 个) ---")
         for i, s in enumerate(reversed(non_zero_lip_samples[-10:])): # 只取后10个
             # 注意：这里的排名是相对于非零样本的排名
             rank_in_gt_zero = n_lip_gt_zero - i
             # 查找在原始完整排序列表中的排名 (可选，但可能有用)
             # original_rank = lip_sorted_samples.index(s) + 1 # 效率不高，如果需要频繁查找可以优化
             logger.info(f"Rank(>0) {rank_in_gt_zero}/{n_lip_gt_zero}. ID: {s['id']}, 平均 Face 占比: {s['face_area_percent']:.4%}, 平均 Lip 占比: {s['lip_area_percent']:.4%}")

         # 计算并记录嘴唇面积(>0)特定百分位的样本
         lip_percentiles_to_log = [1, 20, 40, 60, 80, 99] # 选择的百分位点
         lip_percentile_samples = get_samples_around_percentiles(non_zero_lip_samples, 'lip_area_percent', lip_percentiles_to_log, num_samples=10)
         for p, samples in lip_percentile_samples.items():
             logger.info(f"--- 嘴唇面积占比(>0)约 {p}% 百分位点的 10 个样本 (共 {n_lip_gt_zero} 个占比>0样本) ---")
             if samples:
                 target_rank_gt_zero = int(round(p / 100.0 * (n_lip_gt_zero - 1))) + 1
                 logger.info(f"    (目标排名(>0)约: {target_rank_gt_zero}/{n_lip_gt_zero})")
                 for s in samples:
                     # 注意 rank 是在 non_zero_lip_samples 中的排名
                     logger.info(f"Rank(>0) {s['rank']}/{n_lip_gt_zero}. ID: {s['id']}, 平均 Face 占比: {s['face_area_percent']:.4%}, 平均 Lip 占比: {s['lip_area_percent']:.4%}")
             else:
                 logger.warning(f"    未能获取 {p}% 百分位点的样本(>0)。")
    else:
         logger.warning("数据集中没有找到嘴唇面积占比大于0的有效样本，无法列出最大的10个或计算百分位样本。")

    # --- 保存统计摘要 (百分比) ---
    stats_summary_path = os.path.join(output_dir, "mask_area_percentage_summary.json")
    def safe_stat(data):
        return data.item() if isinstance(data, np.generic) else data

    face_stats = {}
    if len(valid_face_percents) > 0:
        face_stats = {
            "mean_percent": safe_stat(np.mean(valid_face_percents) * 100),
            "median_percent": safe_stat(np.median(valid_face_percents) * 100),
            "min_percent": safe_stat(np.min(valid_face_percents) * 100),
            "max_percent": safe_stat(np.max(valid_face_percents) * 100),
        }

    lip_stats_gt_zero = {}
    if 'non_zero_lip_percents' in locals() and len(non_zero_lip_percents) > 0:
         lip_stats_gt_zero = {
             "count": len(non_zero_lip_percents),
             "mean_percent": safe_stat(np.mean(non_zero_lip_percents) * 100),
             "median_percent": safe_stat(np.median(non_zero_lip_percents) * 100),
             "min_percent": safe_stat(np.min(non_zero_lip_percents) * 100),
             "max_percent": safe_stat(np.max(non_zero_lip_percents) * 100),
         }

    # 获取排序后的极值列表用于保存
    min_face_samples_summary = face_sorted_samples[:10] # 使用已排序列表
    max_face_samples_summary = face_sorted_samples[-10:] # 使用已排序列表
    min_lip_samples_summary = lip_sorted_samples[:10] # 使用已排序列表
    max_lip_samples_gt_zero_summary = non_zero_lip_samples[-10:] if non_zero_lip_samples else [] # 使用已排序列表

    summary_data = {
        "total_samples_in_json": len(data_anno),
        "tasks_created": len(tasks),
        "processed_tasks": processed_count, # Pool 处理的任务数
        "valid_samples_for_stats": len(sample_data), # 成功获取数据的样本数
        "skipped_samples_total": skipped_count, # 包括 json 解析跳过 + 文件不存在跳过
        "error_samples": error_count,
        "face_area_percent_stats": face_stats,
        "lip_area_percent_stats (percent > 0)": lip_stats_gt_zero,
        "samples_with_near_zero_lip_percent": zero_lip_count if 'zero_lip_count' in locals() else 0,
        "top10_min_face_percent": min_face_samples_summary,
        "top10_max_face_percent": max_face_samples_summary,
        "top10_min_lip_percent": min_lip_samples_summary,
        "top10_max_lip_percent_gt_zero": max_lip_samples_gt_zero_summary, # 重命名键更清晰
        "face_percentile_samples": face_percentile_samples, # 添加人脸百分位样本
        "lip_percentile_samples_gt_zero": lip_percentile_samples if non_zero_lip_samples else {} # 添加嘴唇(>0)百分位样本
    }
    try:
        with open(stats_summary_path, 'w') as f:
            # 正确处理 numpy 和其他不可序列化类型
            def json_serializer(obj):
                if isinstance(obj, np.generic):
                    return obj.item()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                raise TypeError(f"Type {type(obj)} not serializable")
            json.dump(summary_data, f, indent=4, default=json_serializer)
        logger.info(f"面积占比统计摘要已保存至: {stats_summary_path}")
    except Exception as e:
        logger.error(f"保存面积占比统计摘要失败: {e}")

    # --- 筛选并保存低于面部 99 百分位的数据 ---
    logger.info("开始筛选面部面积占比低于 99 百分位的样本...")
    filtered_sample_data = [] # 初始化以备后用
    face_99_percentile_threshold = float('inf') # 默认为无穷大，如果没有数据则不过滤
    if len(valid_face_percents) > 0:
        face_99_percentile_threshold = np.percentile(valid_face_percents, 99)
        logger.info(f"计算得到的面部面积 99 百分位阈值: {face_99_percentile_threshold:.4%}")

        # 筛选 sample_data (原始包含所有有效样本的列表)
        filtered_sample_data = [s for s in sample_data if s['face_area_percent'] <= face_99_percentile_threshold]
        num_filtered_out = len(sample_data) - len(filtered_sample_data)
        logger.info(f"筛选完成，保留 {len(filtered_sample_data)} 个样本 (面部面积 <= 99 百分位)，移除了 {num_filtered_out} 个样本。")

        # 保存筛选后的数据到单独的文件 (仅包含处理后的 key)
        filtered_data_path = os.path.join(output_dir, "filtered_mask_data_below_face_99p.json")
        try:
            with open(filtered_data_path, 'w') as f:
                # 使用之前的序列化函数
                json.dump(filtered_sample_data, f, indent=4, default=json_serializer)
            logger.info(f"已将筛选后样本的基本信息保存至: {filtered_data_path}")
        except Exception as e:
            logger.error(f"保存筛选后样本基本信息失败: {e}")
    else:
        logger.warning("没有有效的面部面积数据，无法进行筛选。分桶将基于所有有效样本进行。")
        # 如果没有面部数据，则退化为使用所有有效样本进行分桶
        filtered_sample_data = sample_data
        logger.info(f"使用所有 {len(filtered_sample_data)} 个有效样本进行后续分桶。")

    # --- 基于筛选后的数据进行分桶，并以原始格式保存 ---
    logger.info("开始基于筛选后的数据进行 5 个桶的划分 (按面积数值范围)...")
    if filtered_sample_data: # 确保筛选后仍有数据
        # 1. 获取筛选后样本的面部面积百分比列表
        filtered_face_percents = np.array([s['face_area_percent'] for s in filtered_sample_data])

        if len(filtered_face_percents) > 0:
            # 2. 确定筛选后样本面部面积占比的最小值和最大值
            min_face_percent = np.min(filtered_face_percents)
            max_face_percent = np.max(filtered_face_percents)
            logger.info(f"筛选后面部面积占比范围: [{min_face_percent:.4%}, {max_face_percent:.4%}]")

            # 3. 使用 linspace 计算 5 个桶的面积比例边界 (6 个点)
            # 确保最大值稍微大一点，以包含等于 max_face_percent 的情况
            value_boundaries = np.linspace(min_face_percent, max_face_percent + 1e-9, 6)
            logger.info(f"基于面积数值范围的分桶边界值: {[f'{b:.4%}' for b in value_boundaries]}")

            # 4. 创建 sample_id 到 分桶信息 的映射 (只包含筛选后的样本)
            binned_info_map = {}
            # 定义用于 np.digitize 的边界 (不含第一个和最后一个)
            # right=False: bin[i-1] <= x < bin[i]
            # digitize 返回 0 到 4
            bin_edges = value_boundaries[1:-1] # 使用 b1, b2, b3, b4 作为上边界（不含）

            for sample in filtered_sample_data:
                sample_percent = sample['face_area_percent']
                # 使用 np.digitize 确定桶索引
                # 如果 sample_percent == max_face_percent，它会 >= bin_edges[-1] (b4)，返回 4
                bin_index = np.digitize(sample_percent, bins=bin_edges, right=False)

                binned_info_map[sample['id']] = {
                    'face_area_percent': sample['face_area_percent'],
                    'face_percent_bin': int(bin_index) # 确保是 int
                }

            # 验证每个桶的大小 (基于映射)
            bin_counts = [0] * 5
            for info in binned_info_map.values():
                if 0 <= info['face_percent_bin'] < 5:
                    bin_counts[info['face_percent_bin']] += 1
                else:
                     logger.warning(f"样本 {info['id']} 计算出的桶索引 {info['face_percent_bin']} 超出范围 [0, 4]，请检查逻辑。")
            logger.info(f"每个桶的样本数量 (基于数值范围分桶和映射): {bin_counts}")

            # 5. 遍历原始 data_anno，只保留筛选后数据并添加分桶信息
            final_binned_output = []
            processed_ids_in_output = 0
            skipped_anno_no_path_or_failed = 0
            skipped_anno_above_threshold = 0

            logger.info(f"遍历原始 JSON 数据 (共 {len(data_anno)} 条)，筛选并添加分桶信息...")
            for original_item in data_anno:
                face_emb_path = original_item.get("face_emb_path")
                potential_sample_id = None

                if face_emb_path:
                    temp_file_name = face_emb_path
                    if temp_file_name.endswith(".png"):
                         temp_file_name = temp_file_name.replace(".png", ".pt")
                    potential_sample_id = os.path.splitext(temp_file_name)[0]

                # 检查 ID 是否在我们的分桶映射中 (这意味着它已被处理且低于 99p 阈值)
                if potential_sample_id and potential_sample_id in binned_info_map:
                    output_item = original_item.copy()
                    info = binned_info_map[potential_sample_id]
                    output_item['face_area_percent'] = info['face_area_percent']
                    output_item['face_percent_bin'] = info['face_percent_bin']
                    final_binned_output.append(output_item)
                    processed_ids_in_output += 1
                elif potential_sample_id:
                    # 有 ID 但不在 map 中，说明要么处理失败，要么高于阈值
                    original_sample_info = next((s for s in sample_data if s['id'] == potential_sample_id), None)
                    # 需要检查 face_99_percentile_threshold 是否已定义
                    if 'face_99_percentile_threshold' in locals() and original_sample_info and original_sample_info['face_area_percent'] > face_99_percentile_threshold:
                         skipped_anno_above_threshold += 1
                    else:
                         skipped_anno_no_path_or_failed += 1 # 处理失败或路径无效等
                else:
                     skipped_anno_no_path_or_failed += 1 # 原始路径就没有

            logger.info(f"原始 JSON 遍历完成。")
            logger.info(f"最终输出列表中的条目总数 (低于 99p 且已处理): {len(final_binned_output)}")
            logger.info(f"因面部面积 >= 99p 而被跳过的条目数: {skipped_anno_above_threshold}")
            logger.info(f"因无有效路径或处理失败而被跳过的条目数: {skipped_anno_no_path_or_failed}")

            # 6. 保存最终筛选并分桶的数据
            binned_data_path = os.path.join(output_dir, "binned_mask_data_face_5bins.json")
            try:
                with open(binned_data_path, 'w') as f:
                    # 使用之前的序列化函数
                    json.dump(final_binned_output, f, indent=4, default=json_serializer)
                logger.info(f"已将筛选后 (低于 99p) 并按数值范围分桶的数据 (保留原始格式) 保存至: {binned_data_path}")
            except Exception as e:
                logger.error(f"保存最终筛选并分桶的数据失败: {e}")
        else:
             logger.warning("筛选后没有有效的面部面积数据，无法进行分桶操作和保存。")

    else:
        logger.warning("筛选后没有有效的样本数据，无法进行分桶操作和保存。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用多进程统计人脸和嘴唇掩码的面积占比分布，并可选择性筛选数据")
    parser.add_argument(
        "--json_path",
        type=str,
        default=DEFAULT_JSON_PATH,
        help=f"包含样本信息的 JSON 文件路径 (默认: {DEFAULT_JSON_PATH})"
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        help="存储 .pt 格式掩码坐标文件的目录 (默认: JSON 文件所在目录下的 'face_mask' 子目录)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"保存统计结果和图表的目录 (默认: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"用于处理文件的进程数 (默认: {DEFAULT_NUM_WORKERS})"
    )
    # 移除 --add_tqdm 参数，默认使用 tqdm
    # parser.add_argument(
    #     '--add_tqdm', 
    #     action='store_true', 
    #     help='在主循环中添加 tqdm 进度条'
    # )

    args = parser.parse_args()

    # 如果 mask_dir 未指定，则根据 json_path 推断
    if args.mask_dir is None:
        if os.path.exists(args.json_path) and os.path.isfile(args.json_path):
             args.mask_dir = os.path.join(os.path.dirname(args.json_path), "face_mask")
             logger.info(f"未指定 --mask_dir，将使用推断的路径: {args.mask_dir}")
        else:
             logger.error(f"JSON 文件路径无效 ({args.json_path}) 且未指定 --mask_dir，无法继续。")
             sys.exit(1)

    if not os.path.isdir(args.mask_dir):
        logger.error(f"指定的 Mask 目录不存在或不是一个目录: {args.mask_dir}")
        sys.exit(1)

    # 限制 num_workers 不超过 CPU 核心数，且至少为 1
    args.num_workers = max(1, min(args.num_workers, os.cpu_count()))
    logger.info(f"实际使用的工作进程数: {args.num_workers}")

    # 直接调用，传入 num_workers
    calculate_mask_statistics(args.json_path, args.mask_dir, args.output_dir, args.num_workers)