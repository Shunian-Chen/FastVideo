import json
import os
import shutil
from tqdm import tqdm
import argparse

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="Prepare data for preprocessing.")
# 添加参数定义
parser.add_argument("--original_data_path", type=str, required=True, help="Path to the original data JSON file.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the processed data and copied files.")
parser.add_argument("--original_video_dir", type=str, required=True, help="Original directory containing video and audio files.")
parser.add_argument("--target_video_dir", type=str, required=True, help="Target base directory for video and audio paths in the output JSON.")
parser.add_argument("--caption_data_path", type=str, default=None, help="Optional path to the caption data JSON file.")

# 解析命令行参数
args = parser.parse_args()

# 使用参数值
original_data_path = args.original_data_path
output_dir = args.output_dir
original_video_dir = args.original_video_dir
target_video_dir = args.target_video_dir
caption_data_path = args.caption_data_path

print(f"original_data_path: {original_data_path}")
print(f"output_dir: {output_dir}")
print(f"original_video_dir: {original_video_dir}")
print(f"target_video_dir: {target_video_dir}")
print(f"caption_data_path: {caption_data_path}")

original_data = json.load(open(original_data_path, "r"))
# 如果提供了 caption_data_path，则加载并合并数据
# 注意：这部分合并逻辑可能需要根据你的 caption_data 格式进行调整或确认
if caption_data_path and os.path.exists(caption_data_path):
    caption_data = json.load(open(caption_data_path, "r"))
    print(f"caption_data: {len(caption_data)}")
    caption_data_dict = {data["id"]: data for data in caption_data}
    print(f"caption_data_dict: {len(caption_data_dict)}")


    merged_data_list = []
    for data in original_data:
        if data["id"] in caption_data_dict:
            merge_item = data
            # 假设 caption 数据结构如原注释所示
            merge_item['caption'] = caption_data_dict[data["id"]]["description"] if "description" in caption_data_dict[data["id"]] else 'The man is talking'
        else:
            merge_item['caption'] = 'The man is talking'
        merged_data_list.append(merge_item)
    original_data = merged_data_list # 使用合并后的数据
    print(f"merged_data: {len(original_data)}")
else:
    print("Caption data path not provided or file not found. Skipping caption merging.")

video_dir = os.path.join(output_dir, "video")
os.makedirs(video_dir, exist_ok=True)
audio_dir = os.path.join(output_dir, "original_audio")
os.makedirs(audio_dir, exist_ok=True)

processed_data = []
for data in tqdm(original_data):
    video_path_source = data["video-path"].replace(original_video_dir, target_video_dir)
    video_name = os.path.basename(video_path_source)
    video_output_path = os.path.join(video_dir, video_name)

    audio_path_source = data['audio-path'].replace(original_video_dir, target_video_dir)
    audio_name = os.path.basename(audio_path_source)
    audio_output_path = os.path.join(audio_dir, audio_name)

    if os.path.exists(video_output_path):
        # print(f"video_output_path: {video_output_path} already exists")
        pass
    elif os.path.exists(video_path_source):
        shutil.copy(video_path_source, video_output_path)
    else:
        print(f"Warning: Source video file not found: {video_path_source}")
        continue

    if os.path.exists(audio_output_path):
        # print(f"audio_output_path: {audio_output_path} already exists")
        pass
    elif os.path.exists(audio_path_source):
        shutil.copy(audio_path_source, audio_output_path)
    else:
        print(f"Warning: Source audio file not found: {audio_path_source}")
        continue
    processed_video_path = video_name
    processed_audio_path = audio_name

    processed_data.append({
        "path": processed_video_path,
        "audio_path": processed_audio_path,
        "resolution": {
            "width": data["width"],
            "height": data["height"]
        },
        "fps": data["fps"],
        "duration": float(data["durations"].replace("s", "")),
        "cap": [data['caption'] if 'caption' in data else '']
    })

with open(os.path.join(output_dir, "data.json"), "w") as f:
    json.dump(processed_data, f, indent=4)



