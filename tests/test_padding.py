import pytest
import torch

def test_audio_padding():
    # 测试用例1: 需要padding的情况
    bs, w, b, d = 2, 3, 4, 5
    original_frames = 5
    sp_size = 3
    
    # 创建模拟数据
    audio_emb = torch.ones((bs, original_frames, w, b, d))
    print(f"audio_emb: {audio_emb}")
    class NcclInfo:
        sp_size = 3
    nccl_info = NcclInfo()
    
    # 执行padding
    audio_frame = audio_emb.shape[1]
    if audio_frame % nccl_info.sp_size != 0:
        remainder = audio_frame % nccl_info.sp_size
        padding = nccl_info.sp_size - remainder
        pad_shape = list(audio_emb.shape)
        pad_shape[1] = padding
        zeros_pad = torch.zeros(pad_shape, dtype=audio_emb.dtype, device=audio_emb.device)
        audio_emb = torch.cat([audio_emb, zeros_pad], dim=1)
    print(f"audio_emb: {audio_emb}")
    # 验证结果
    assert audio_emb.shape[1] == 6  # 5 + 1 padding
    assert torch.all(audio_emb[:, :original_frames] == 1)  # 原始数据保留
    assert torch.all(audio_emb[:, original_frames:] == 0)  # padding部分为0

    # 测试用例2: 不需要padding的情况
    original_frames = 6
    audio_emb = torch.ones((bs, original_frames, w, b, d))
    original_data = audio_emb.clone()
    
    audio_frame = audio_emb.shape[1]
    if audio_frame % nccl_info.sp_size != 0:
        # 这里不应该执行
        raise AssertionError("不应该进入这个分支")
    
    assert audio_emb.shape[1] == original_frames
    assert torch.all(audio_emb == original_data)

if __name__ == "__main__":
    test_audio_padding()
