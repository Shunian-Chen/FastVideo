import cv2
import json
import os
from glob import glob

def extract_first_frames(video_dir, output_dir, jsonl_path):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有mp4视频文件
    video_paths = glob(os.path.join(video_dir, "*.mp4"))
    
    with open(jsonl_path, 'w') as f:
        for video_path in video_paths:
            # 读取视频第一帧
            cap = cv2.VideoCapture(video_path)
            success, frame = cap.read()
            if not success:
                continue  # 跳过无法读取的视频
            cap.release()
            
            # 生成输出路径
            base_name = os.path.basename(video_path).rsplit('.', 1)[0]
            image_path = os.path.join(output_dir, f"{base_name}_first_frame.jpg")
            
            # 保存图片
            cv2.imwrite(image_path, frame)
            
            # 写入JSONL
            record = {
                "prompt": "The man is talking",
                "video_path": video_path,
                "visual_input": image_path
            }
            f.write(json.dumps(record) + '\n')

# 使用示例
if __name__ == "__main__":
    extract_first_frames(
        video_dir="/path/to/videos",
        output_dir="/path/to/output_images",
        jsonl_path="/path/to/output.jsonl"
    )
