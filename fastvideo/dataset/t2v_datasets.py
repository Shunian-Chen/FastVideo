import json
import math
import os
import random
import logging
from collections import Counter
from os.path import join as opj
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset

from fastvideo.utils.dataset_utils import DecordInit
from fastvideo.utils.logging_ import main_print

# 配置日志记录
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class DataSetProg(metaclass=SingletonMeta):

    def __init__(self):
        self.cap_list = []
        self.elements = []
        self.num_workers = 1
        self.n_elements = 0
        self.worker_elements = dict()
        self.n_used_elements = dict()
        self.processed_samples = []  # 新增：存储已处理的样本信息

    def set_cap_list(self, num_workers, cap_list, n_elements):
        self.num_workers = num_workers
        self.cap_list = cap_list
        self.n_elements = n_elements
        self.elements = list(range(n_elements))
        random.shuffle(self.elements)
        print(f"n_elements: {len(self.elements)}", flush=True)

        for i in range(self.num_workers):
            self.n_used_elements[i] = 0
            per_worker = int(
                math.ceil(len(self.elements) / float(self.num_workers)))
            start = i * per_worker
            end = min(start + per_worker, len(self.elements))
            self.worker_elements[i] = self.elements[start:end]

    def get_item(self, work_info):
        if work_info is None:
            worker_id = 0
        else:
            worker_id = work_info.id

        idx = self.worker_elements[worker_id][
            self.n_used_elements[worker_id] %
            len(self.worker_elements[worker_id])]
        self.n_used_elements[worker_id] += 1
        return idx


dataset_prog = DataSetProg()


def filter_resolution(h,
                      w,
                      max_h_div_w_ratio=17 / 16,
                      min_h_div_w_ratio=8 / 16):
    if h / w <= max_h_div_w_ratio and h / w >= min_h_div_w_ratio:
        return True
    return False


class T2V_dataset(Dataset):

    def __init__(self, args, transform, temporal_sample, tokenizer,
                 transform_topcrop,
                 video_processor = None
                ):
        self.gpu_rank = args.gpu_rank   # new
        self.video_processor = args.video_processor  # new
        self.data = args.data_merge_path
        self.num_frames = args.num_frames
        self.train_fps = args.train_fps
        self.use_image_num = args.use_image_num
        self.transform = transform
        self.transform_topcrop = transform_topcrop
        self.temporal_sample = temporal_sample
        self.tokenizer = tokenizer
        self.text_max_length = args.text_max_length
        self.cfg = args.cfg
        self.output_dir = args.output_dir
        self.speed_factor = args.speed_factor
        self.max_height = args.max_height
        self.max_width = args.max_width
        self.drop_short_ratio = args.drop_short_ratio
        assert self.speed_factor >= 1
        self.v_decoder = DecordInit()
        self.video_length_tolerance_range = args.video_length_tolerance_range
        self.support_Chinese = True
        if "mt5" not in args.text_encoder_name:
            self.support_Chinese = False

        cap_list = self.get_cap_list()

        assert len(cap_list) > 0
        cap_list, self.sample_num_frames = self.define_frame_index(cap_list)
        self.lengths = self.sample_num_frames

        n_elements = len(cap_list)
        dataset_prog.set_cap_list(args.dataloader_num_workers, cap_list,
                                  n_elements)

        print(f"video length: {len(dataset_prog.cap_list)}", flush=True)

    
    def get_video_v2(self, idx, json_file="frame_indices_shunian.json"):
        video_path = dataset_prog.cap_list[idx]["path"]
        assert os.path.exists(video_path), f"file {video_path} do not exist!"
        
        # 检查样本是否已处理
        if dataset_prog.cap_list[idx].get("is_processed", False):
            logging.info(f"[Rank {self.gpu_rank}] 样本 {video_path} 已处理，跳过处理")
            # 返回一个特殊标记，表示该样本已处理
            return dict(
                pixel_values=torch.tensor([]),
                text=dataset_prog.cap_list[idx]["cap"],
                input_ids=torch.tensor([]),
                cond_mask=torch.tensor([]),
                path=video_path,
                timestamp=torch.tensor([-1]),  # 使用-1表示已处理
                frames=torch.tensor([]),
                face_mask_path=dataset_prog.cap_list[idx].get("face_mask_path", ""),
                face_emb_path=dataset_prog.cap_list[idx].get("face_emb_path", ""),
                audio_emb_path=dataset_prog.cap_list[idx].get("audio_emb_path", ""),
                is_processed=True  # 添加标记
            )
        
        frame_indices = dataset_prog.cap_list[idx]["sample_frame_index"]
        try:
            assert len(frame_indices) == self.num_frames, f"frame_indices length is not equal to self.num_frames"
        except AssertionError as e:
            logging.error(f"[Rank {self.gpu_rank}] AssertionError for video {video_path}: {e}. frame_indices length: {len(frame_indices)}, expected: {self.num_frames}")
            # 可以选择返回失败字典或继续（取决于是否允许帧数不匹配）
            # import ipdb; ipdb.set_trace() # 调试时使用
            # 返回失败字典以跳过此样本
            return dict(
                pixel_values=torch.tensor([]),
                text="",
                input_ids=torch.tensor([]),
                cond_mask=torch.tensor([]),
                path=video_path,
                timestamp=torch.tensor([]),
                frames=torch.tensor(0),
                face_mask_path="",
                face_emb_path="",
                audio_emb_path="",
                is_processed=False # 标记未处理
            )
            # raise e # 如果希望程序停止

        ## ------new code------
        if self.video_processor is not None or True: # 注意：这里的 or True 会导致此分支始终执行
            from pathlib import Path
            timestamp, face_mask_path, face_emb_path, audio_emb_path = self.video_processor.process(Path(video_path), self.transform,
                                         frame_indices)
            if timestamp == -1:
                # video_processor 处理失败，返回结构一致的空字典
                logging.warning(f"[Rank {self.gpu_rank}] video_processor failed for video: {video_path}. Skipping.")
                return dict(
                    pixel_values=torch.tensor([]),
                    text="", # 使用空字符串
                    input_ids=torch.tensor([]),
                    cond_mask=torch.tensor([]),
                    path=video_path, # 返回路径字符串
                    timestamp=torch.tensor([]),
                    frames=torch.tensor(0), # 返回 0 帧
                    face_mask_path="", # 使用空字符串
                    face_emb_path="", # 使用空字符串
                    audio_emb_path="", # 使用空字符串
                    is_processed=False # 标记未处理
                )
            # 确保路径是字符串而不是Path对象
            face_mask_path = str(face_mask_path) if face_mask_path is not None else "" # 使用空字符串代替 None
            face_emb_path = str(face_emb_path) if face_emb_path is not None else "" # 使用空字符串代替 None
            face_emb_path = str(face_emb_path) if face_emb_path is not None else "" # 使用空字符串代替 None
        ## ------end-----------
        
        try: # 添加 try-except 块来捕获视频读取错误
            torchvision_video, _, metadata = torchvision.io.read_video(
                video_path, pts_unit='sec', output_format="TCHW") # 建议明确 pts_unit
        except Exception as e:
            logging.error(f"[Rank {self.gpu_rank}] Error reading video {video_path}: {e}. Skipping.")
            # 视频读取失败，返回结构一致的空字典
            return dict(
                pixel_values=torch.tensor([]),
                text="",
                input_ids=torch.tensor([]),
                cond_mask=torch.tensor([]),
                path=video_path,
                timestamp=torch.tensor([]),
                frames=torch.tensor(0),
                face_mask_path="",
                face_emb_path="",
                audio_emb_path="",
                is_processed=False # 标记未处理
            )

        # --- 新增检查 ---
        if torchvision_video.shape[0] == 0:
            logging.warning(f"[Rank {self.gpu_rank}] Failed to read video or video is empty: {video_path}. Skipping.")
            # 视频为空，返回结构一致的空字典
            return dict(
                pixel_values=torch.tensor([]),
                text="",
                input_ids=torch.tensor([]),
                cond_mask=torch.tensor([]),
                path=video_path,
                timestamp=torch.tensor([]),
                frames=torch.tensor(0),
                face_mask_path="",
                face_emb_path="",
                audio_emb_path="",
                is_processed=False # 标记未处理
            )
        # --- 结束新增检查 ---

        try: # 添加 try-except 块来捕获索引错误 (虽然前面的检查应该能避免，但以防万一)
            video = torchvision_video[frame_indices]
        except IndexError as e:
            logging.error(f"[Rank {self.gpu_rank}] IndexError when accessing frames for video {video_path}: {e}. Video shape: {torchvision_video.shape}, Indices length: {len(frame_indices)}. Skipping.")
             # 索引失败，返回结构一致的空字典
            return dict(
                pixel_values=torch.tensor([]),
                text="",
                input_ids=torch.tensor([]),
                cond_mask=torch.tensor([]),
                path=video_path,
                timestamp=torch.tensor([]),
                frames=torch.tensor(0),
                face_mask_path="",
                face_emb_path="",
                audio_emb_path="",
                is_processed=False # 标记未处理
            )
        except Exception as e: # 捕获其他可能的错误
            logging.error(f"[Rank {self.gpu_rank}] Unexpected error processing video {video_path} after reading: {e}. Skipping.")
            return dict(
                pixel_values=torch.tensor([]),
                text="",
                input_ids=torch.tensor([]),
                cond_mask=torch.tensor([]),
                path=video_path,
                timestamp=torch.tensor([]),
                frames=torch.tensor(0),
                face_mask_path="",
                face_emb_path="",
                audio_emb_path="",
                is_processed=False # 标记未处理
            )

        video = self.transform(video)
        video = rearrange(video, "t c h w -> c t h w")
        video = video.to(torch.uint8)
        assert video.dtype == torch.uint8

        h, w = video.shape[-2:]
        assert (
            h / w <= 17 / 16 and h / w >= 8 / 16
        ), f"Only videos with a ratio (h/w) less than 17/16 and more than 8/16 are supported. But video ({video_path}) found ratio is {round(h / w, 2)} with the shape of {video.shape}"

        video = video.float() / 127.5 - 1.0

        text = dataset_prog.cap_list[idx]["cap"]
        if not isinstance(text, list):
            text = [text]
        text = [random.choice(text)]

        text = text[0] if random.random() > self.cfg else ""
        
        text_tokens_and_mask = self.tokenizer(
            text,
            max_length=self.text_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        input_ids = text_tokens_and_mask["input_ids"]
        cond_mask = text_tokens_and_mask["attention_mask"]
        return dict(
            pixel_values=video,
            text=text,
            input_ids=input_ids,
            cond_mask=cond_mask,
            path=video_path,
            timestamp=timestamp,        # new
            frames=len(frame_indices),   # new
            face_mask_path=face_mask_path,  # new
            face_emb_path=face_emb_path,    # new
            audio_emb_path=audio_emb_path,  # new
            is_processed=False # 标记未处理 (因为这是实时处理的)
        )
        
    def set_checkpoint(self, n_used_elements):
        for i in range(len(dataset_prog.n_used_elements)):
            dataset_prog.n_used_elements[i] = n_used_elements

    def __len__(self):
        return dataset_prog.n_elements

    def __getitem__(self, idx):

        data = self.get_data(idx)
        return data

    def get_data(self, idx):
        path = dataset_prog.cap_list[idx]["path"]
        if path.endswith(".mp4"):
            # return self.get_video(idx)
            return self.get_video_v2(idx)
        else:
            return self.get_image(idx)

    def get_video(self, idx):
        video_path = dataset_prog.cap_list[idx]["path"]
        assert os.path.exists(video_path), f"file {video_path} do not exist!"
        frame_indices = dataset_prog.cap_list[idx]["sample_frame_index"]
        torchvision_video, _, metadata = torchvision.io.read_video(
            video_path, output_format="TCHW")
        video = torchvision_video[frame_indices]
        video = self.transform(video)
        video = rearrange(video, "t c h w -> c t h w")
        video = video.to(torch.uint8)
        assert video.dtype == torch.uint8

        h, w = video.shape[-2:]
        assert (
            h / w <= 17 / 16 and h / w >= 8 / 16
        ), f"Only videos with a ratio (h/w) less than 17/16 and more than 8/16 are supported. But video ({video_path}) found ratio is {round(h / w, 2)} with the shape of {video.shape}"

        video = video.float() / 127.5 - 1.0

        text = dataset_prog.cap_list[idx]["cap"]
        if not isinstance(text, list):
            text = [text]
        text = [random.choice(text)]

        text = text[0] if random.random() > self.cfg else ""
        text_tokens_and_mask = self.tokenizer(
            text,
            max_length=self.text_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = text_tokens_and_mask["input_ids"]
        cond_mask = text_tokens_and_mask["attention_mask"]
        return dict(
            pixel_values=video,
            text=text,
            input_ids=input_ids,
            cond_mask=cond_mask,
            path=video_path,
        )

    def get_image(self, idx):
        image_data = dataset_prog.cap_list[
            idx]  # [{'path': path, 'cap': cap}, ...]

        image = Image.open(image_data["path"]).convert("RGB")  # [h, w, c]
        image = torch.from_numpy(np.array(image))  # [h, w, c]
        image = rearrange(image, "h w c -> c h w").unsqueeze(0)  #  [1 c h w]
        # for i in image:
        #     h, w = i.shape[-2:]
        #     assert h / w <= 17 / 16 and h / w >= 8 / 16, f'Only image with a ratio (h/w) less than 17/16 and more than 8/16 are supported. But found ratio is {round(h / w, 2)} with the shape of {i.shape}'

        image = (self.transform_topcrop(image) if "human_images"
                 in image_data["path"] else self.transform(image)
                 )  #  [1 C H W] -> num_img [1 C H W]
        image = image.transpose(0, 1)  # [1 C H W] -> [C 1 H W]

        image = image.float() / 127.5 - 1.0

        caps = (image_data["cap"] if isinstance(image_data["cap"], list) else
                [image_data["cap"]])
        caps = [random.choice(caps)]
        text = caps
        input_ids, cond_mask = [], []
        text = text[0] if random.random() > self.cfg else ""
        text_tokens_and_mask = self.tokenizer(
            text,
            max_length=self.text_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = text_tokens_and_mask["input_ids"]  # 1, l
        cond_mask = text_tokens_and_mask["attention_mask"]  # 1, l
        return dict(
            pixel_values=image,
            text=text,
            input_ids=input_ids,
            cond_mask=cond_mask,
            path=image_data["path"],
        )

    def define_frame_index(self, cap_list):
        new_cap_list = []
        sample_num_frames = []
        cnt_too_long = 0
        cnt_too_short = 0
        cnt_no_cap = 0
        cnt_no_resolution = 0
        cnt_resolution_mismatch = 0
        cnt_movie = 0
        cnt_img = 0
        for i in cap_list:
            path = i["path"]
            cap = i.get("cap", None)
            # ======no caption=====
            if cap is None:
                cnt_no_cap += 1
                continue
            if path.endswith(".mp4"):
                # ======no fps and duration=====
                duration = i.get("duration", None)
                fps = i.get("fps", None)
                if fps is None or duration is None:
                    continue

                # ======resolution mismatch=====
                resolution = i.get("resolution", None)
                if resolution is None:
                    cnt_no_resolution += 1
                    continue
                else:
                    if (resolution.get("height", None) is None
                            or resolution.get("width", None) is None):
                        cnt_no_resolution += 1
                        continue
                    height, width = i["resolution"]["height"], i["resolution"][
                        "width"]
                    aspect = self.max_height / self.max_width
                    hw_aspect_thr = 1.5
                    is_pick = filter_resolution(
                        height,
                        width,
                        max_h_div_w_ratio=hw_aspect_thr * aspect,
                        min_h_div_w_ratio=1 / hw_aspect_thr * aspect,
                    )
                    if not is_pick:
                        # print("resolution mismatch")
                        cnt_resolution_mismatch += 1
                        continue

                # import ipdb;ipdb.set_trace()
                i["num_frames"] = math.ceil(fps * duration)
                # max 5.0 and min 1.0 are just thresholds to filter some videos which have suitable duration.
                if i["num_frames"] / fps > self.video_length_tolerance_range * (
                        self.num_frames / self.train_fps * self.speed_factor
                ):  # too long video is not suitable for this training stage (self.num_frames)
                    cnt_too_long += 1
                    continue

                # resample in case high fps, such as 50/60/90/144 -> train_fps(e.g, 24)
                frame_interval = fps / self.train_fps
                start_frame_idx = 0
                frame_indices = np.arange(start_frame_idx, i["num_frames"],
                                          frame_interval).astype(int)

                # comment out it to enable dynamic frames training
                if (len(frame_indices) < self.num_frames
                        and random.random() < self.drop_short_ratio):
                    cnt_too_short += 1
                    continue

                #  too long video will be temporal-crop randomly
                if len(frame_indices) > self.num_frames:
                    begin_index, end_index = self.temporal_sample(
                        len(frame_indices))
                    frame_indices = frame_indices[begin_index:end_index]
                    if len(frame_indices) != self.num_frames:
                        print("error")
                    # frame_indices = frame_indices[:self.num_frames]  # head crop
                i["sample_frame_index"] = frame_indices.tolist()
                new_cap_list.append(i)
                i["sample_num_frames"] = len(
                    i["sample_frame_index"]
                )  # will use in dataloader(group sampler)
                sample_num_frames.append(i["sample_num_frames"])
            elif path.endswith(".jpg"):  # image
                cnt_img += 1
                new_cap_list.append(i)
                i["sample_num_frames"] = 1
                sample_num_frames.append(i["sample_num_frames"])
            else:
                raise NameError(
                    f"Unknown file extension {path.split('.')[-1]}, only support .mp4 for video and .jpg for image"
                )
        # import ipdb;ipdb.set_trace()
        main_print(
            f"no_cap: {cnt_no_cap}, too_long: {cnt_too_long}, too_short: {cnt_too_short}, "
            f"no_resolution: {cnt_no_resolution}, resolution_mismatch: {cnt_resolution_mismatch}, "
            f"Counter(sample_num_frames): {Counter(sample_num_frames)}, cnt_movie: {cnt_movie}, cnt_img: {cnt_img}, "
            f"before filter: {len(cap_list)}, after filter: {len(new_cap_list)}"
        )
        return new_cap_list, sample_num_frames

    def decord_read(self, path, frame_indices):
        decord_vr = self.v_decoder(path)
        video_data = decord_vr.get_batch(frame_indices).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1,
                                        2)  # (T, H, W, C) -> (T C H W)
        return video_data

    def read_jsons(self, data):
        cap_lists = []
        with open(data, "r") as f:
            folder_anno = [
                i.strip().split(",") for i in f.readlines()
                if len(i.strip()) > 0
            ]
        # print(folder_anno)
        for folder, anno in folder_anno:
            with open(anno, "r") as f:
                sub_list = json.load(f)
            for i in range(len(sub_list)):
                sub_list[i]["path"] = opj(folder, sub_list[i]["path"])
            cap_lists += sub_list
        return cap_lists

    def get_cap_list(self):
        cap_lists = self.read_jsons(self.data)
        
        # 如果video_processor存在，则检查并标记已处理的样本
        if hasattr(self, 'video_processor') and self.video_processor is not None:
            output_dir = self.video_processor.output_dir
            processed_count = 0
            
            logging.info(f"[Rank {self.gpu_rank}] 开始检查已处理的样本，原始样本数量: {len(cap_lists)}")
            
            for item in tqdm(cap_lists, desc=f"[Rank {self.gpu_rank}] 检查已处理的样本"):
                video_path = item["path"]
                video_name = os.path.basename(video_path).split(".")[0]
                
                # 构建所有需要检查的文件路径
                latent_path = os.path.join(output_dir, "latent", f"{video_name}.pt")
                
                # 获取base_dir（根据setup_directories函数的逻辑）
                video_path_obj = Path(video_path)
                base_dir = self.output_dir
                
                face_mask_path = os.path.join(base_dir, "face_mask", f"{video_name}.png")
                face_emb_path = os.path.join(base_dir, "face_emb", f"{video_name}.pt")
                audio_emb_path = os.path.join(base_dir, "audio_emb", f"{video_name}.pt")
                
                # 检查所有文件是否都存在
                latent_exists = os.path.exists(latent_path)
                # face_mask_exists = os.path.exists(face_mask_path)
                # face_emb_exists = os.path.exists(face_emb_path)
                audio_emb_exists = os.path.exists(audio_emb_path)
                
                # all_exist = (latent_exists and face_mask_exists and 
                            # face_emb_exists and audio_emb_exists)
                all_exist = latent_exists and audio_emb_exists
                # 如果所有文件都存在，则标记该样本为已处理
                if all_exist:
                    processed_count += 1
                    # 标记样本为已处理
                    item["is_processed"] = True
                    # 记录相关文件路径
                    # latent = torch.load(latent_path)
                    # length = latent.shape[1]
                    item["latent_path"] = latent_path
                    # item['length'] = length
                    item["face_mask_path"] = face_mask_path
                    item["face_emb_path"] = face_emb_path
                    item["audio_emb_path"] = audio_emb_path
                    
                    # 将已处理的样本添加到processed_samples列表中
                    processed_sample = {
                        "latent_path": f"{video_name}.pt",
                        # "length": length,
                        "caption": item["cap"],
                        "face_mask_path": face_mask_path,
                        "face_emb_path": face_emb_path,
                        "audio_emb_path": audio_emb_path
                    }
                    dataset_prog.processed_samples.append(processed_sample)
                    
                    if processed_count % 1000 == 0 and self.gpu_rank == 0:
                        logging.info(f"[Rank {self.gpu_rank}] 已标记 {processed_count} 个已处理的样本")
                else:
                    # 记录缺失的文件
                    missing_files = []
                    if not latent_exists:
                        missing_files.append("latent")
                    # if not face_mask_exists:
                    #     missing_files.append("face_mask")
                    # if not face_emb_exists:
                    #     missing_files.append("face_emb")
                    if not audio_emb_exists:
                        missing_files.append("audio_emb")
                    
                    if self.gpu_rank == 0 and len(missing_files) > 0 and len(missing_files) % 100 == 0:
                        logging.info(f"[Rank {self.gpu_rank}] 视频 {video_name} 缺少文件: {', '.join(missing_files)}")
            
            logging.info(f"[Rank {self.gpu_rank}] 检查完成: 已标记 {processed_count} 个已处理的样本，总样本数量: {len(cap_lists)}")
            return cap_lists
        
        return cap_lists
