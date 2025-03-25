## hallo3
import logging
import shutil
import os
from pathlib import Path
import cv2
import torch
import numpy as np
import subprocess
from datetime import datetime
import time
from typing import Dict, List, Tuple

from .sgm.utils.audio_processor import AudioProcessor
from .sgm.utils.image_processor import ImageProcessorForDataProcessing
from .sgm.utils.util import convert_video_to_images, extract_audio_from_videos, get_fps

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def setup_directories(video_path: Path, output_dir: Path) -> dict:
    """
    Setup directories for storing processed files.

    Args:
        video_path (Path): Path to the video file.

    Returns:
        dict: A dictionary containing paths for various directories.
    """
    base_dir = output_dir
    dirs = {
        "face_mask": base_dir / "face_mask",
        "face_emb": base_dir / "face_emb",
        "audio_emb": base_dir / "audio_emb"
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


class VideoProcessor:
    def __init__(self, args,
                 output_dir: Path, 
                 image_processor: ImageProcessorForDataProcessing, 
                 audio_processor: AudioProcessor,
                 ):
        self.output_dir = output_dir
        self.image_processor = image_processor
        self.audio_processor = audio_processor
        self.max_height = args.max_height
        self.max_width = args.max_width
        # self.transform = transform  # 依赖注入

    def process(self, video_path: Path,
             transform, 
             frame_indices: List[int]
             ) -> None:
        assert video_path.exists(), f"Video path {video_path} does not exist"
        dirs = setup_directories(video_path, self.output_dir)
        logging.info(f"Processing video: {video_path}")

        try:
            # 首先检查是否已经存在所有必要的文件
            face_mask_path = dirs["face_mask"] / f"{video_path.stem}.png"
            face_emb_path = dirs["face_emb"] / f"{video_path.stem}.pt"
            audio_emb_path = dirs["audio_emb"] / f"{video_path.stem}.pt"
            
            # 如果所有文件都已存在，直接返回
            if face_mask_path.exists() and face_emb_path.exists() and audio_emb_path.exists():
                logging.info(f"所有必要文件已存在，跳过处理: {video_path.stem}")
                timestamp = datetime.now().strftime("%m-%d-%H")
                return timestamp, face_mask_path, face_emb_path, audio_emb_path
            
            timestamp = datetime.now().strftime("%m-%d-%H")
            new_video_path = self.output_dir / 'videos' / f"{video_path.stem}.mp4"
            new_video_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 获取原始视频的fps
            try:
                fps = get_fps(video_path)
            except Exception as e:
                logging.warning(f"获取视频FPS失败: {e}，使用默认值25")
                fps = 25.0
            
            # 检查frame_indices是否为空
            if not frame_indices:
                logging.error(f"Empty frame_indices for video: {video_path}")
                return timestamp, None, None, None
            
            # 计算音频裁切的起止时间
            start_frame = min(frame_indices)
            end_frame = max(frame_indices)
            start_time = start_frame / fps
            end_time = (end_frame + 1) / fps
            
            # 检查是否需要处理人脸相关文件
            need_process_face = not (face_mask_path.exists() and face_emb_path.exists())
            # 检查是否需要处理音频文件
            need_process_audio = not audio_emb_path.exists()
            
            # 如果需要处理任何一种文件，则需要创建新视频
            if need_process_face or need_process_audio:
                try:
                    create_new_video(video_path, new_video_path, frame_indices)
                    logging.info(f"New video created with frames {frame_indices}: {new_video_path}")
                except Exception as e:
                    logging.error(f"创建新视频失败: {e}")
                    return timestamp, None, None, None
            
            # 处理人脸相关文件
            if need_process_face:
                images_output_dir = self.output_dir / 'images' / video_path.stem
                images_output_dir.mkdir(parents=True, exist_ok=True)
                try:
                    # 清空目录，避免之前处理失败的残留文件
                    for file in images_output_dir.glob('*'):
                        file.unlink()
                        
                    images_output_dir = convert_video_to_images(
                        new_video_path, images_output_dir)
                    logging.info(f"Images saved to: {images_output_dir}")
                    
                    # 验证图像是否成功生成
                    image_files = list(images_output_dir.glob('*.jpg'))
                    if not image_files:
                        raise Exception("未能生成任何图像文件")
                        
                except Exception as e:
                    logging.error(f"视频转换为图像失败: {e}")
                    # 创建空的掩码和嵌入向量
                    empty_mask = np.zeros((self.max_height, self.max_width), dtype=np.uint8)
                    cv2.imwrite(str(face_mask_path), empty_mask)
                    empty_emb = torch.zeros(512, dtype=torch.float32)
                    torch.save(empty_emb, str(face_emb_path))
                    logging.warning(f"创建了空的人脸掩码和嵌入向量: {face_mask_path}, {face_emb_path}")
                    
                    if need_process_audio:
                        return timestamp, face_mask_path, face_emb_path, None
                    else:
                        return timestamp, face_mask_path, face_emb_path, audio_emb_path
                
                # 计时
                start_time_proc = time.time()
                try:
                    face_mask, face_emb, _, _, _ = self.image_processor.preprocess(
                        images_output_dir)
                    elapsed_time = time.time() - start_time_proc
                    logging.info(f"[time] Image preprocessing completed in {elapsed_time:.2f} seconds")
                except Exception as e:
                    logging.error(f"图像预处理失败: {e}")
                    # 创建空的掩码和嵌入向量
                    face_mask = None
                    face_emb = None
                
                ## ------new code------
                # 检查face_mask是否为None，如果是则创建一个空的掩码
                if face_mask is None:
                    logging.warning(f"No face detected in video: {video_path.stem}, creating empty face mask")
                    # 获取视频的第一帧以确定尺寸
                    try:
                        first_frame_path = next((p for p in images_output_dir.glob('*.jpg') if os.path.exists(p)), None)
                        if first_frame_path:
                            try:
                                first_frame = cv2.imread(str(first_frame_path))
                                if first_frame is not None and first_frame.size > 0:
                                    height, width = first_frame.shape[:2]
                                    face_mask = np.zeros((height, width), dtype=np.uint8)
                                else:
                                    # 如果无法读取图像，使用默认尺寸
                                    face_mask = np.zeros((self.max_height, self.max_width), dtype=np.uint8)
                            except Exception as e:
                                logging.error(f"读取第一帧失败: {e}")
                                face_mask = np.zeros((self.max_height, self.max_width), dtype=np.uint8)
                        else:
                            # 如果没有找到帧，使用默认尺寸
                            face_mask = np.zeros((self.max_height, self.max_width), dtype=np.uint8)
                    except Exception as e:
                        logging.error(f"获取第一帧路径失败: {e}")
                        face_mask = np.zeros((self.max_height, self.max_width), dtype=np.uint8)
                
                try:
                    face_mask = torch.from_numpy(face_mask).unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)
                    face_mask = transform(face_mask)                        # crop and resize
                    face_mask = face_mask.squeeze().cpu().numpy()           # (H, W)
                    face_mask = (face_mask > 0).astype(np.uint8) * 255      # 将插值产生的中间过渡值转换255
                except Exception as e:
                    logging.error(f"处理face_mask失败: {e}")
                    # 如果处理失败，创建一个默认大小的空掩码
                    face_mask = np.zeros((224, 224), dtype=np.uint8)  # 假设transform后的大小是224x224
                ## ------end-----------
                
                if images_output_dir.exists():
                    try:
                        shutil.rmtree(images_output_dir)
                        # logging.info(f"Deleted temporary directory: {images_output_dir}")
                    except Exception as e:
                        logging.warning(f"删除临时目录失败: {e}")
            
                try:
                    cv2.imwrite(str(face_mask_path), face_mask)
                    logging.info(f"Face mask saved to: {face_mask_path}")
                except Exception as e:
                    logging.error(f"保存人脸掩码失败: {e}")
                    # 尝试再次创建一个空掩码并保存
                    try:
                        empty_mask = np.zeros((self.max_height, self.max_width), dtype=np.uint8)
                        cv2.imwrite(str(face_mask_path), empty_mask)
                        logging.warning(f"创建并保存了空的人脸掩码: {face_mask_path}")
                    except Exception as e2:
                        logging.error(f"保存空的人脸掩码也失败: {e2}")
                
                # 检查face_emb是否为None，如果是则创建一个空的嵌入向量
                if face_emb is None:
                    logging.warning(f"No face embedding detected in video: {video_path.stem}, creating empty face embedding")
                    face_emb = torch.zeros(512, dtype=torch.float32)  # 假设嵌入向量的维度是512
                
                try:
                    torch.save(face_emb, str(face_emb_path))
                    logging.info(f"Face embedding saved to: {face_emb_path}")
                except Exception as e:
                    logging.error(f"保存人脸嵌入向量失败: {e}")
                    # 尝试再次创建一个空的嵌入向量并保存
                    try:
                        empty_emb = torch.zeros(512, dtype=torch.float32)
                        torch.save(empty_emb, str(face_emb_path))
                        logging.warning(f"创建并保存了空的人脸嵌入向量: {face_emb_path}")
                    except Exception as e2:
                        logging.error(f"保存空的人脸嵌入向量也失败: {e2}")
            else:
                logging.info(f"人脸相关文件已存在，跳过处理: {face_mask_path}, {face_emb_path}")
            
            # 处理音频文件
            if need_process_audio:
                # 检查视频是否有音频轨道
                has_audio = has_audio_stream(new_video_path)
                
                # 设置音频输出路径
                audio_output_dir = self.output_dir / 'audios'
                audio_output_dir.mkdir(parents=True, exist_ok=True)
                audio_output_path = audio_output_dir / f'{new_video_path.stem}.wav'
                
                # 检查是否存在与原始视频同名的音频文件（用于没有音轨的视频）
                original_audio_path = self.output_dir / 'original_audio' / f'{new_video_path.stem}.m4a'
                
                # 如果视频有音频轨道，则从视频中提取音频
                if has_audio:
                    # 检查音频文件是否已存在
                    if audio_output_path.exists():
                        logging.info(f"音频文件已存在，跳过提取: {audio_output_path}")
                    else:
                        try:
                            audio_output_path = extract_audio_from_videos(
                                new_video_path, audio_output_path)
                            logging.info(f"Audio extracted from video to: {audio_output_path}")
                        except Exception as e:
                            logging.error(f"从视频提取音频失败: {e}")
                            return timestamp, face_mask_path, face_emb_path, None
                # 如果视频没有音频轨道，但存在同名音频文件，则裁切该音频文件
                elif original_audio_path.exists():
                    logging.info(f"Video has no audio stream, but found audio file: {original_audio_path}")
                    # 裁切音频
                    try:
                        audio_output_path = trim_audio_file(
                            original_audio_path, audio_output_path, start_time, end_time)
                        logging.info(f"Audio trimmed and saved to: {audio_output_path}")
                    except Exception as e:
                        logging.error(f"裁切音频失败: {e}")
                        return timestamp, face_mask_path, face_emb_path, None
                else:
                    logging.warning(f"Video {new_video_path} has no audio stream and no audio file found at {original_audio_path}.")
                    return timestamp, face_mask_path, face_emb_path, None
                
                # 处理音频嵌入 - 检查音频文件是否存在
                if audio_output_path.exists():
                    # 计时
                    start_time_proc = time.time()
                    try:
                        audio_emb, _ = self.audio_processor.preprocess(audio_output_path, fps=24.0, target_length=len(frame_indices))
                        # audio_emb, _ = self.audio_processor.preprocess(audio_output_path, fps=fps)
                        elapsed_time = time.time() - start_time_proc
                        logging.info(f"[time] Audio preprocessing completed in {elapsed_time:.2f} seconds")    
                        
                        torch.save(audio_emb, str(audio_emb_path))
                        logging.info(f"Audio embedding saved to: {audio_emb_path}")
                    except Exception as e:
                        logging.error(f"音频预处理或保存失败: {e}")
                        return timestamp, face_mask_path, face_emb_path, None
                else:
                    logging.warning(f"No audio file found at {audio_output_path}. Creating empty audio embedding.")
                    return timestamp, face_mask_path, face_emb_path, None
            else:
                logging.info(f"音频嵌入文件已存在，跳过处理: {audio_emb_path}")
            
            # 清理临时视频文件
            if new_video_path.exists() and not (need_process_face or need_process_audio):
                try:
                    os.remove(new_video_path)
                    logging.info(f"删除临时视频文件: {new_video_path}")
                except Exception as e:
                    logging.warning(f"删除临时视频文件失败: {e}")
                
            return timestamp, face_mask_path, face_emb_path, audio_emb_path
        except Exception as e:
            logging.error(f"Failed to process video {video_path}: {e}")
            # 返回None而不是抛出异常，以便处理下一个视频
            return datetime.now().strftime("%m-%d-%H"), None, None, None


def create_new_video(original_video_path, new_video_path, frame_indices):
    """
    基于FFmpeg的select/atrim滤镜，精准截取视频帧与对应音频段。
    :param original_video_path: 原始视频路径
    :param new_video_path:      生成新视频存放路径
    :param frame_indices:       要截取的帧号列表 (如 [1,2,3,4])，须为连续帧。
    """
    fps = get_fps(original_video_path)
    
    start_frame = min(frame_indices)
    end_frame   = max(frame_indices)
    
    ##   根据帧号和 fps 计算音频对应的起止时间
    #    假设我们要包含 end_frame 本身，所以 end_time = (end_frame + 1) / fps
    start_time = start_frame / fps
    end_time   = (end_frame + 1) / fps
    
    # 首先检查视频是否有音频轨道
    cmd_check_audio = [
        "ffprobe", "-v", "error", 
        "-select_streams", "a:0", 
        "-show_entries", "stream=codec_type", 
        "-of", "default=nw=1:nk=1", 
        str(original_video_path)
    ]
    
    try:
        result = subprocess.run(cmd_check_audio, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        has_audio = result.stdout.decode().strip() == "audio"
    except Exception as e:
        logging.warning(f"Error checking audio stream: {e}")
        has_audio = False
    
    # 根据是否有音频轨道构建不同的滤镜命令
    if has_audio:
        # 视频和音频都处理
        filter_complex = (
            f"[0:v]select='between(n,{start_frame},{end_frame})',"
            f"setpts=N/FRAME_RATE/TB[v];"
            f"[0:a]atrim={start_time}:{end_time},asetpts=N/SR/TB[a]"
        )
        map_options = ["-map", "[v]", "-map", "[a]"]
    else:
        # 只处理视频
        filter_complex = (
            f"[0:v]select='between(n,{start_frame},{end_frame})',"
            f"setpts=N/FRAME_RATE/TB[v]"
        )
        map_options = ["-map", "[v]"]
    
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-i", str(original_video_path),
        "-filter_complex", filter_complex
    ] + map_options + [
        "-c:v", "libx264"
    ]
    
    # 只有在有音频的情况下才添加音频编码器
    if has_audio:
        cmd.extend(["-c:a", "aac"])
    
    cmd.append(str(new_video_path))

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error creating new video: {e.stderr.decode()}")
        raise e


def has_audio_stream(video_path):
    """检查视频是否有音频轨道"""
    cmd = [
        "ffprobe", "-v", "error", 
        "-select_streams", "a:0", 
        "-show_entries", "stream=codec_type", 
        "-of", "default=nw=1:nk=1", 
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode().strip() == "audio"
    except Exception as e:
        logging.warning(f"Error checking audio stream: {e}")
        return False


def trim_audio_file(input_audio_path, output_audio_path, start_time, end_time):
    """
    裁切音频文件，使其与视频的起止时间匹配
    
    Args:
        input_audio_path: 输入音频文件路径
        output_audio_path: 输出音频文件路径
        start_time: 开始时间（秒）
        end_time: 结束时间（秒）
    
    Returns:
        输出音频文件路径
    """
    # 检查输出文件是否已存在
    if os.path.exists(output_audio_path):
        logging.info(f"音频文件已存在，跳过裁切: {output_audio_path}")
        return output_audio_path
        
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-i", str(input_audio_path),
        "-ss", f"{start_time:.6f}",
        "-to", f"{end_time:.6f}",
        "-c:a", "pcm_s16le",  # 使用无损编码
        str(output_audio_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info(f"Audio trimmed from {start_time:.2f}s to {end_time:.2f}s")
        return output_audio_path
    except subprocess.CalledProcessError as e:
        logging.error(f"Error trimming audio: {e.stderr.decode()}")
        raise e


## 原脚本2，问题是帧与时间戳可能略有偏差
# def create_new_video(original_video_path, new_video_path, frame_indices, fps):
#     start_time = frame_indices[0] / fps
#     duration = (frame_indices[-1] - frame_indices[0] + 1) / fps

#     cmd = [
#         'ffmpeg', '-y', '-i', str(original_video_path),
#         '-ss', f'{start_time:.2f}', '-t', f'{duration:.2f}',
#         '-c:v', 'libx264', '-c:a', 'aac',
#         str(new_video_path)
#     ]

#     try:
#         subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     except subprocess.CalledProcessError as e:
#         logging.error(f"Error creating new video: {e.stderr.decode()}")
        
        
## 原脚本1，问题是无法保留视频的音频信息
# def create_new_video(original_video_path, new_video_path, frame_indices):
#     """从原视频中提取指定帧并生成新视频"""
#     cap = cv2.VideoCapture(str(original_video_path))
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
#     out = cv2.VideoWriter(str(new_video_path), fourcc, fps, (width, height))
    
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     for i in range(frame_count):
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if i in frame_indices:
#             out.write(frame)
    
#     cap.release()
#     out.release()

