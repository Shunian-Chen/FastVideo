import argparse
import json
import os

import torch
import torch.distributed as dist
import torch.utils.data.dataloader
from accelerate import PartialState
from accelerate.logging import get_logger
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from fastvideo.utils.load import load_text_encoder, load_vae

logger = get_logger(__name__)


class T5dataset(Dataset):

    def __init__(
        self,
        json_path,
        output_dir,
        vae_debug,
    ):
        self.json_path = json_path
        self.output_dir = output_dir
        self.vae_debug = vae_debug
        with open(self.json_path, "r") as f:
            # train_dataset = json.load(f)[1000:2000]
            train_dataset = json.load(f)
            self.train_dataset = sorted(train_dataset,
                                        key=lambda x: x["latent_path"])

    def __getitem__(self, idx):
        item_data = self.train_dataset[idx]
        caption = item_data["caption"][0]
        filename = item_data["latent_path"].split(".")[0].split("/")[-1]
        latent_path = item_data["latent_path"]
        audio_emb_path = item_data["audio_emb_path"]
        face_emb_path = item_data["face_emb_path"]
        full_latent_path = os.path.join(self.output_dir, "latent", latent_path)

        try:
            latent = torch.load(full_latent_path, map_location="cpu")
            length = latent.shape[1]

            result = dict(
                caption=caption,
                filename=filename,
                length=length,
                latent_path=latent_path,
                audio_emb_path=audio_emb_path,
                face_emb_path=face_emb_path,
                error=False
            )

        except RuntimeError as e:
            print(f"[DataLoader Worker] Skipping sample {idx} due to RuntimeError loading latent '{full_latent_path}': {e}")
            result = {"error": True, "latent_path": latent_path}
        except FileNotFoundError:
            print(f"[DataLoader Worker] Skipping sample {idx} because latent file not found: '{full_latent_path}'")
            result = {"error": True, "latent_path": latent_path}
        except Exception as e:
            print(f"[DataLoader Worker] Skipping sample {idx} due to unexpected error loading latent '{full_latent_path}': {e}")
            result = {"error": True, "latent_path": latent_path}

        return result

    def __len__(self):
        return len(self.train_dataset)


def collate_fn(batch):
    """
    Filters out items that have errors and collates the valid ones.
    """
    valid_batch = [item for item in batch if not item.get("error")]
    if not valid_batch:
        # Return None if the entire batch consists of errors
        return None
    # Use the default collate function for the valid items
    return torch.utils.data.dataloader.default_collate(valid_batch)


def main(args):
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size, "local rank", local_rank)

    # Initialize PartialState early to enable logging
    PartialState()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl",
                                init_method="env://",
                                world_size=world_size,
                                rank=local_rank)

    videoprocessor = VideoProcessor(vae_scale_factor=8)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "video"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "latent"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "prompt_embed"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "prompt_attention_mask"),
                exist_ok=True)

    latents_json_path = os.path.join(args.output_dir,
                                     "videos2caption_temp.json")
    train_dataset = T5dataset(latents_json_path, args.output_dir, args.vae_debug)
    text_encoder = load_text_encoder(args.model_type,
                                     args.model_path,
                                     device=device)
    vae, autocast_type, fps = load_vae(args.model_type, args.model_path)
    vae.enable_tiling()
    sampler = DistributedSampler(train_dataset,
                                 rank=local_rank,
                                 num_replicas=world_size,
                                 shuffle=True)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
    )

    json_data = []
    total_batches = len(train_dataloader)
    for _, data in tqdm(enumerate(train_dataloader), disable=local_rank != 0, total = total_batches, desc=f"[Rank {local_rank}] 处理进度", leave=True):
        if data is None:
            logger.warning(f"Rank {local_rank}: Skipping a batch because all samples failed to load.")
            continue

        with torch.inference_mode():

            for idx_in_batch in range(len(data["filename"])):
                video_name = data["filename"][idx_in_batch]
                current_latent_path = data["latent_path"][idx_in_batch]

                # Define output paths
                prompt_embed_path = os.path.join(args.output_dir, "prompt_embed", video_name + ".pt")
                prompt_attention_mask_path = os.path.join(args.output_dir, "prompt_attention_mask", video_name + ".pt")

                # Check if output files already exist
                if os.path.exists(prompt_embed_path) and os.path.exists(prompt_attention_mask_path):
                    logger.info(f"Rank {local_rank}: Skipping {video_name} as embeddings already exist.")
                    item = {}
                    item["length"] = int(data["length"][idx_in_batch])
                    item["latent_path"] = os.path.basename(data["latent_path"][idx_in_batch])
                    item["audio_emb_path"] = os.path.basename(data["audio_emb_path"][idx_in_batch])
                    item["face_emb_path"] = os.path.basename(data["face_emb_path"][idx_in_batch])
                    item["prompt_embed_path"] = os.path.basename(prompt_embed_path)
                    item["prompt_attention_mask"] = os.path.basename(prompt_attention_mask_path)
                    item["caption"] = data["caption"][idx_in_batch]
                    json_data.append(item)
                    print(f"Rank {local_rank}: Skipping {video_name} as embeddings already exist.")
                    continue # Skip to the next item in the batch

                with torch.autocast("cuda", dtype=autocast_type):
                    prompt_embeds, prompt_attention_mask = text_encoder.encode_prompt(
                        prompt=data["caption"],
                    )
                if args.vae_debug:
                    try:
                        full_latent_path = os.path.join(args.output_dir, "latent", current_latent_path)
                        latents = torch.load(full_latent_path, map_location="cpu")
                        latents = latents.to(device)

                        video_frames = vae.decode(latents.unsqueeze(0), return_dict=False)[0]
                        video = videoprocessor.postprocess_video(video_frames)

                        video_output_path = os.path.join(args.output_dir, "video", video_name + ".mp4")
                        export_to_video(video, video_output_path, fps=fps)
                        print(f"sample videovideo {video_name} saved")
                    except Exception as e:
                        logger.error(f"Rank {local_rank}: Error during VAE decode/save for {video_name} (latent: {current_latent_path}): {e}")

                # Save the embedding and mask for the current item
                # prompt_embeds and prompt_attention_mask are batched, index with idx_in_batch
                torch.save(prompt_embeds[idx_in_batch], prompt_embed_path)
                torch.save(prompt_attention_mask[idx_in_batch], prompt_attention_mask_path)

                item = {}
                item["length"] = int(data["length"][idx_in_batch])
                item["latent_path"] = os.path.basename(data["latent_path"][idx_in_batch])
                item["audio_emb_path"] = os.path.basename(data["audio_emb_path"][idx_in_batch])
                item["face_emb_path"] = os.path.basename(data["face_emb_path"][idx_in_batch])
                item["prompt_embed_path"] = os.path.basename(prompt_embed_path)
                item["prompt_attention_mask"] = os.path.basename(prompt_attention_mask_path)
                item["caption"] = data["caption"][idx_in_batch]
                json_data.append(item)

    dist.barrier()
    local_data = json_data
    gathered_data = [None] * world_size
    dist.all_gather_object(gathered_data, local_data)
    if local_rank == 0:
        # os.remove(latents_json_path)
        all_json_data = [item for sublist in gathered_data for item in sublist]
        with open(os.path.join(args.output_dir, "videos2caption.json"),
                  "w") as f:
            json.dump(all_json_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument("--model_path", type=str, default="data/mochi")
    parser.add_argument("--model_type", type=str, default="mochi")
    # text encoder & vae & diffusion model
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help=
        "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--text_encoder_name",
                        type=str,
                        default="google/t5-v1_1-xxl")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=
        "The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--vae_debug", action="store_true")
    args = parser.parse_args()
    main(args)
