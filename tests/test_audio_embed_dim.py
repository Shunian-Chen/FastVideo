import json
import torch
import os
with open("/data/nas/yexin/workspace/shunian/model_training/FastVideo/data/50_hour_test_480p_48frames/videos2caption.json", "r") as f:
    latent_embeds = json.load(f)

home_dir = "/data/nas/yexin/workspace/shunian/model_training/FastVideo/data/50_hour_test_480p_49frames"
audio_emb_dir = os.path.join(home_dir, "audio_emb")

for item in latent_embeds:

    audio_emb_path = os.path.join(audio_emb_dir, item["audio_emb_path"])
    if not os.path.exists(audio_emb_path):
        # print(f"audio_emb_path {audio_emb_path} does not exist")
        continue
    audio_emb = torch.load(audio_emb_path)
    print(audio_emb.shape)


