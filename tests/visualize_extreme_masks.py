import cv2
import torch
import json
import os
import numpy as np
from loguru import logger
import sys
import argparse
from tqdm import tqdm
import multiprocessing
from functools import partial

# --- 默认配置 ---
DEFAULT_SUMMARY_JSON = "/data/nas/yexin/workspace/shunian/model_training/FastVideo/tests/mask_statistics_output/mask_area_percentage_summary.json"
# 注意：根据 mask_area_summary.json 的来源，确定正确的视频和 mask 目录
DEFAULT_VIDEO_DIR = "/data/nas/yexin/workspace/shunian/model_training/FastVideo/data/252_hour_test_480p_49frames/videos"
DEFAULT_MASK_DIR = "/data/nas/yexin/workspace/shunian/model_training/FastVideo/data/200_hour_480p_49frames_eng/face_mask"
DEFAULT_OUTPUT_DIR = "model_training/FastVideo/tests/extreme_mask_visualizations_percentage_percentile"
DEFAULT_NUM_WORKERS = 16 # 默认使用 CPU 核心数

# --- 可视化参数 ---
FACE_COLOR = (0, 0, 255)  # BGR for Red
LIP_COLOR = (0, 255, 0)   # BGR for Green
ALPHA = 0.4               # 透明度 (0=完全透明, 1=完全不透明)

# --- 日志配置 ---
# 在主进程中配置，子进程默认继承
logger.remove()
logger.add(sys.stdout, level="INFO")

def draw_transparent_rect(img, rect_coords, color, alpha):
    """在图像上绘制半透明矩形"""
    x1, y1, x2, y2 = [int(c) for c in rect_coords]
    if x2 <= x1 or y2 <= y1: # 无效矩形
        return img

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)  # -1 表示填充矩形
    img_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return img_new

def process_single_video(task_args):
    """处理单个视频的可视化任务 (用于多进程)。"""
    sample_id, video_path, mask_path, output_video_path, group_name = task_args

    if os.path.exists(output_video_path):
        # logger.debug(f"输出文件已存在，跳过: {output_video_path}") # 不在子进程打日志
        return {"status": "skipped_exists", "id": sample_id, "group": group_name}

    if not os.path.exists(video_path):
        # logger.error(f"视频文件未找到: {video_path}...")
        return {"status": "error_video_not_found", "id": sample_id, "path": video_path, "group": group_name}

    if not os.path.exists(mask_path):
        # logger.error(f"Mask 坐标文件未找到: {mask_path}...")
        return {"status": "error_mask_not_found", "id": sample_id, "path": mask_path, "group": group_name}

    # 加载 Mask 坐标
    try:
        mask_data = torch.load(mask_path, map_location="cpu")
        if not (isinstance(mask_data, dict) and "original_height" in mask_data and "frames" in mask_data):
            # logger.error(f"Mask 文件格式不正确: {mask_path}...")
            return {"status": "error_mask_format", "id": sample_id, "path": mask_path, "group": group_name}
        frame_coords_list = mask_data["frames"]
        num_mask_frames = len(frame_coords_list)
    except Exception as e:
        # logger.error(f"加载 Mask 文件时出错 {mask_path}: {e}...")
        return {"status": "error_mask_load", "id": sample_id, "path": mask_path, "error": str(e), "group": group_name}

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # logger.error(f"无法打开视频文件: {video_path}...")
        return {"status": "error_video_open", "id": sample_id, "path": video_path, "group": group_name}

    # 获取视频属性
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 检查 mask 帧数和视频帧数
    frame_mismatch = False
    if abs(num_mask_frames - num_video_frames) > 5:
        # logger.warning(f"{sample_id} (分组 {group_name}): 视频帧数 ({num_video_frames}) 与 Mask 帧数 ({num_mask_frames}) 不匹配超过阈值。按较少者处理。")
        frame_mismatch = True # 可以在结果中标记这个警告
    max_frames_to_process = min(num_video_frames, num_mask_frames)

    # 设置视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        # logger.error(f"无法创建视频写入器: {output_video_path}...")
        cap.release()
        return {"status": "error_video_write_create", "id": sample_id, "path": output_video_path, "group": group_name}

    # logger.debug(f"开始处理并写入视频: {output_video_path}") # 主进程处理日志
    frame_idx = 0
    success = True
    try:
        while success and frame_idx < max_frames_to_process:
            success, frame = cap.read()
            if not success:
                break

            coords = None
            if frame_idx < len(frame_coords_list) and frame_coords_list[frame_idx] is not None:
                 if len(frame_coords_list[frame_idx]) == 8:
                     coords = frame_coords_list[frame_idx]

            processed_frame = frame.copy()
            if coords:
                fx1, fy1, fx2, fy2, lx1, ly1, lx2, ly2 = [int(c) for c in coords]
                fx1, fx2 = max(0, fx1), min(frame_width, fx2)
                fy1, fy2 = max(0, fy1), min(frame_height, fy2)
                lx1, lx2 = max(0, lx1), min(frame_width, lx2)
                ly1, ly2 = max(0, ly1), min(frame_height, ly2)

                if fx2 > fx1 and fy2 > fy1:
                    processed_frame = draw_transparent_rect(processed_frame, (fx1, fy1, fx2, fy2), FACE_COLOR, ALPHA)
                if lx2 > lx1 and ly2 > ly1:
                    processed_frame = draw_transparent_rect(processed_frame, (lx1, ly1, lx2, ly2), LIP_COLOR, ALPHA)

            out.write(processed_frame)
            frame_idx += 1

        # logger.debug(f"完成写入视频 ({frame_idx} 帧): {output_video_path}")
        return {"status": "success", "id": sample_id, "frames_written": frame_idx, "frame_mismatch_warn": frame_mismatch, "group": group_name, "output_path": output_video_path}

    except Exception as e:
        # logger.error(f"处理视频帧时发生错误 {output_video_path}: {e}", exc_info=True)
        # 尝试删除可能已部分写入的文件
        try:
            if os.path.exists(output_video_path):
                os.remove(output_video_path)
        except OSError:
            pass # 忽略删除错误
        return {"status": "error_frame_processing", "id": sample_id, "error": str(e), "group": group_name}
    finally:
        cap.release()
        out.release()

def visualize_masks_on_video(summary_json_path, video_dir, mask_dir, output_dir, num_workers):
    """读取 summary json，使用多进程为极值和百分位样本可视化 mask 到视频上，并按类别保存。"""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"根输出目录: {os.path.abspath(output_dir)}")
    logger.info(f"读取统计摘要: {summary_json_path}")
    logger.info(f"视频源目录: {video_dir}")
    logger.info(f"Mask 坐标目录: {mask_dir}")
    logger.info(f"使用 {num_workers} 个工作进程。")

    try:
        with open(summary_json_path, 'r') as f:
            summary_data = json.load(f)
        logger.info("成功加载统计摘要 JSON。")
    except FileNotFoundError:
        logger.error(f"统计摘要 JSON 文件未找到: {summary_json_path}")
        return
    except json.JSONDecodeError:
        logger.error(f"无法解析统计摘要 JSON 文件: {summary_json_path}")
        return

    # --- 整理需要处理的样本和分组 ---
    tasks = []
    samples_by_group = {}

    def add_samples(group_name, sample_list):
        nonlocal tasks
        group_output_dir = os.path.join(output_dir, group_name)
        os.makedirs(group_output_dir, exist_ok=True)
        if sample_list:
            samples_by_group[group_name] = sample_list # 保留分组信息用于总结
            logger.info(f"找到 {len(sample_list)} 个样本用于分组 '{group_name}'。")
            for sample_item in sample_list:
                sample_id = sample_item.get('id')
                if not sample_id:
                     logger.warning(f"分组 {group_name} 中发现缺少 'id' 的样本项: {sample_item}，跳过添加到任务列表。")
                     continue
                video_filename = f"{sample_id}.mp4"
                mask_filename = f"{sample_id}.pt"
                video_path = os.path.join(video_dir, video_filename)
                mask_path = os.path.join(mask_dir, mask_filename)
                output_video_path = os.path.join(group_output_dir, f"{sample_id}_visualization.mp4")
                tasks.append((sample_id, video_path, mask_path, output_video_path, group_name))
        else:
            logger.warning(f"未在 JSON 中找到或样本列表为空: '{group_name}'。")

    add_samples("face_min_top10", summary_data.get("top10_min_face_percent", []))
    add_samples("face_max_top10", summary_data.get("top10_max_face_percent", []))
    add_samples("lip_min_top10", summary_data.get("top10_min_lip_percent", []))
    add_samples("lip_max_top10_gt_zero", summary_data.get("top10_max_lip_percent_gt_zero", []))

    face_percentiles = summary_data.get("face_percentile_samples", {})
    for p, samples in face_percentiles.items():
        add_samples(f"face_percentile_{p}", samples)

    lip_percentiles = summary_data.get("lip_percentile_samples_gt_zero", {})
    for p, samples in lip_percentiles.items():
        add_samples(f"lip_percentile_{p}_gt_zero", samples)

    if not tasks:
        logger.error("未能构建任何处理任务，无法继续。")
        return

    logger.info(f"总计需要处理 {len(tasks)} 个任务 (一个样本可能生成多个视频，如果属于多个分组)。")

    # --- 多进程处理 ---
    processed_count = 0
    error_count = 0
    skipped_count = 0
    frame_mismatch_warnings = 0
    error_details = []

    logger.info("开始多进程处理视频可视化任务...")
    with multiprocessing.Pool(processes=num_workers) as pool:
        # 使用 imap_unordered 获取结果，便于更新进度条
        results_iterator = pool.imap_unordered(process_single_video, tasks)
        for result in tqdm(results_iterator, total=len(tasks), desc="处理视频"):
            status = result.get("status", "unknown")
            sample_id = result.get("id", "未知")
            group_name = result.get("group", "未知分组")

            if status == "success":
                processed_count += 1
                logger.debug(f"成功处理: {sample_id} (分组: {group_name}), 输出: {result.get('output_path')}")
                if result.get("frame_mismatch_warn", False):
                    frame_mismatch_warnings += 1
                    logger.warning(f"帧数不匹配警告: {sample_id} (分组: {group_name})")
            elif status == "skipped_exists":
                skipped_count += 1
                logger.debug(f"跳过已存在: {sample_id} (分组: {group_name})")
            elif status.startswith("error_"):
                error_count += 1
                error_msg = f"错误 ({status}): {sample_id} (分组: {group_name})"
                if "path" in result:
                    error_msg += f", 文件: {result['path']}"
                if "error" in result:
                    error_msg += f", 原因: {result['error']}"
                logger.error(error_msg)
                error_details.append(result) # 记录详细错误信息
            else:
                error_count += 1
                logger.error(f"未知处理结果状态 '{status}' 来自样本 {sample_id} (分组: {group_name}): {result}")
                error_details.append(result)

    logger.info(f"--- 可视化完成 ---")
    logger.info(f"成功生成视频数: {processed_count}")
    logger.info(f"跳过已存在视频数: {skipped_count}")
    logger.info(f"发生错误的视频数: {error_count}")
    if frame_mismatch_warnings > 0:
        logger.warning(f"出现帧数不匹配警告的视频数: {frame_mismatch_warnings}")
    logger.info(f"可视化结果保存在根目录: {output_dir}")

    # 可以选择性地保存详细错误日志
    if error_details:
        error_log_path = os.path.join(output_dir, "visualization_errors.json")
        try:
            with open(error_log_path, 'w') as f:
                json.dump(error_details, f, indent=4)
            logger.info(f"详细错误信息已保存至: {error_log_path}")
        except Exception as e:
            logger.error(f"保存错误日志失败: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用多进程在视频上可视化面积极端和特定百分位样本的 Face 和 Lip Mask，并按分组保存")
    parser.add_argument(
        "--summary_json",
        type=str,
        default=DEFAULT_SUMMARY_JSON,
        help=f"包含极值样本 ID 的统计摘要 JSON 文件路径 (默认: {DEFAULT_SUMMARY_JSON})"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=DEFAULT_VIDEO_DIR,
        help=f"包含原始 MP4 视频文件的目录 (默认: {DEFAULT_VIDEO_DIR})"
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default=DEFAULT_MASK_DIR,
        help=f"存储 .pt 格式掩码坐标文件的目录 (默认: {DEFAULT_MASK_DIR})"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"保存可视化结果视频的目录 (默认: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None, # 默认值设为 None，后面再决定
        help=f"用于处理视频的进程数 (默认: CPU 核心数)"
    )

    args = parser.parse_args()

    # 决定工作进程数
    if args.num_workers is None:
        args.num_workers = os.cpu_count() # 使用所有核心
    args.num_workers = max(1, args.num_workers) # 确保至少为 1

    if not os.path.exists(args.summary_json):
        logger.error(f"指定的统计摘要 JSON 文件不存在: {args.summary_json}")
        sys.exit(1)
    if not os.path.isdir(args.video_dir):
        logger.error(f"指定的视频目录不存在或不是一个目录: {args.video_dir}")
        sys.exit(1)
    if not os.path.isdir(args.mask_dir):
        logger.error(f"指定的 Mask 目录不存在或不是一个目录: {args.mask_dir}")
        sys.exit(1)

    visualize_masks_on_video(args.summary_json, args.video_dir, args.mask_dir, args.output_dir, args.num_workers) 