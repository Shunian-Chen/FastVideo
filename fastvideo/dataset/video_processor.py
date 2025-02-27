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

from .sgm.utils.audio_processor import AudioProcessor
from .sgm.utils.image_processor import ImageProcessorForDataProcessing
from .sgm.utils.util import convert_video_to_images, extract_audio_from_videos, get_fps

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def setup_directories(video_path: Path) -> dict:
    """
    Setup directories for storing processed files.

    Args:
        video_path (Path): Path to the video file.

    Returns:
        dict: A dictionary containing paths for various directories.
    """
    base_dir = video_path.parent.parent
    dirs = {
        "face_mask": base_dir / "face_mask",
        "face_emb": base_dir / "face_emb",
        "audio_emb": base_dir / "audio_emb"
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


class VideoProcessor:
    def __init__(self, output_dir: Path, 
                 image_processor: ImageProcessorForDataProcessing, 
                 audio_processor: AudioProcessor,
                 ):
        self.output_dir = output_dir
        self.image_processor = image_processor
        self.audio_processor = audio_processor
        # self.transform = transform  # 依赖注入

    def process(self, video_path: Path,
             transform, 
             frame_indices: list[int]
             ) -> None:
        assert video_path.exists(), f"Video path {video_path} does not exist"
        dirs = setup_directories(video_path)
        logging.info(f"Processing video: {video_path}")

        try:
            timestamp = datetime.now().strftime("%m-%d-%H")
            new_video_path = self.output_dir / 'videos' / f"{video_path.stem}.mp4"
            new_video_path.parent.mkdir(parents=True, exist_ok=True)
            create_new_video(video_path, new_video_path, frame_indices)
            logging.info(f"New video created with frames {frame_indices}: {new_video_path}")
            
            images_output_dir = self.output_dir / 'images' / video_path.stem
            images_output_dir.mkdir(parents=True, exist_ok=True)
            images_output_dir = convert_video_to_images(
                new_video_path, images_output_dir)
            logging.info(f"Images saved to: {images_output_dir}")
            
            fps = get_fps(new_video_path)

            audio_output_dir = self.output_dir / 'audios'
            audio_output_dir.mkdir(parents=True, exist_ok=True)
            audio_output_path = audio_output_dir / f'{new_video_path.stem}.wav'
            audio_output_path = extract_audio_from_videos(
                new_video_path, audio_output_path)
            logging.info(f"Audio extracted to: {audio_output_path}")

            # 计时
            start_time = time.time()
            face_mask, face_emb, _, _, _ = self.image_processor.preprocess(
                images_output_dir)
            elapsed_time = time.time() - start_time
            logging.info(f"[time] Image preprocessing completed in {elapsed_time:.2f} seconds")
            
            ## ------new code------
            face_mask = torch.from_numpy(face_mask).unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)
            face_mask = transform(face_mask)                        # crop and resize
            face_mask = face_mask.squeeze().cpu().numpy()           # (H, W)
            face_mask = (face_mask > 0).astype(np.uint8) * 255      # 将插值产生的中间过渡值转换255
            ## ------end-----------
            
            if images_output_dir.exists():
                shutil.rmtree(images_output_dir)
                # shutil.rmtree(new_video_path)
                # logging.info(f"Deleted temporary directory: {images_output_dir}")
        
            cv2.imwrite(
                str(dirs["face_mask"] / f"{video_path.stem}.png"), face_mask)
            torch.save(face_emb, str(
                dirs["face_emb"] / f"{video_path.stem}.pt"))
            audio_path = self.output_dir / "audios" / f"{new_video_path.stem}.wav"
            
            # 计时
            start_time = time.time()
            audio_emb, _ = self.audio_processor.preprocess(audio_path, fps=24.0)
            # audio_emb, _ = self.audio_processor.preprocess(audio_path, fps=fps)
            elapsed_time = time.time() - start_time
            logging.info(f"[time] Audio preprocessing completed in {elapsed_time:.2f} seconds")    
            
            torch.save(audio_emb, str(
                dirs["audio_emb"] / f"{new_video_path.stem}.pt"))
            return timestamp
        except Exception as e:
            logging.error(f"Failed to process video {video_path}: {e}")


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
    
    ##  构造滤镜，分别处理视频和音频
    #   - 视频: 只保留 [start_frame, end_frame] 的帧
    #   - 音频: 截取 [start_time, end_time] 这段音频
    #   这里需要注意：FFmpeg 的 between(n,a,b) 是闭区间包含 a 和 b
    filter_complex = (
        f"[0:v]select='between(n,{start_frame},{end_frame})',"
        f"setpts=N/FRAME_RATE/TB[v];"
        f"[0:a]atrim={start_time}:{end_time},asetpts=N/SR/TB[a]"
    )
    
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-i", str(original_video_path),
        "-filter_complex", filter_complex,
        "-map", "[v]",  # 映射处理后的视频
        "-map", "[a]",  # 映射处理后的音频
        "-c:v", "libx264",  # 可自行调整视频编码器
        "-c:a", "aac",      # 可自行调整音频编码器
        str(new_video_path)
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error creating new video: {e.stderr.decode()}")
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

