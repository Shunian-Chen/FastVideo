import json
import os
import shutil
from tqdm import tqdm

original_data_path = "/wangbenyou/shunian/workspace/talking_face/video_processing/video_filtering/sample_diverse_data/data_sampled.json"
original_data = json.load(open(original_data_path, "r"))
caption_data_path = "/wangbenyou/shunian/workspace/talking_face/Talking-Face-Datapipe/7_video_caption/data/test_time/sample/fixed_sample_data_total.json"
caption_data = json.load(open(caption_data_path, "r"))
output_dir = "/wangbenyou/shunian/workspace/talking_face/model_training/FastVideo/data/50_hour_test"


print(f"original_data: {len(original_data)}")
original_data_dict = {data["id"]: data for data in original_data}
print(f"original_data_dict: {len(original_data_dict)}")

merged_data = []
for data in caption_data:
    if data["video_folder"] in original_data_dict:
        merge_item = original_data_dict[data["video_folder"]]
        merge_item['caption'] = data["response"]['choices'][0]['message']['content']["description_summary"]
        merged_data.append(merge_item)

print(f"merged_data: {len(merged_data)}")



video_dir = os.path.join(output_dir, "video")
os.makedirs(video_dir, exist_ok=True)
audio_dir = os.path.join(output_dir, "audio")
os.makedirs(audio_dir, exist_ok=True)



processed_data = []
for data in tqdm(merged_data):
    video_path = data["video-path"]
    video_path = video_path.split("/")[-1]
    video_output_path = os.path.join(video_dir, video_path)
    audio_path = data['audio-path']
    audio_path = audio_path.split("/")[-1]
    audio_output_path = os.path.join(audio_dir, audio_path)

    # shutil.copy(data["video-path"].replace("/wangbenyou/shunian", "/sds_wangby/datasets_dir/datasets/shunian/"), video_output_path)
    # shutil.copy(data['audio-path'].replace("/wangbenyou/shunian", "/sds_wangby/datasets_dir/datasets/shunian/"), audio_output_path)


    processed_data.append({
        "path": video_path,
        "audio_path": audio_path,
        "resolution": {
            "width": data["width"],
            "height": data["height"]
        },
        "fps": data["fps"],
        "duration": float(data["durations"].replace("s", "")),
        "cap": [data['caption']]
    })


with open(os.path.join(output_dir, "data.json"), "w") as f:
    json.dump(processed_data, f, indent=4)



