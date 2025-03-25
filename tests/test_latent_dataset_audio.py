import os
import json
import torch
import pytest
from fastvideo.dataset.latent_datasets_audio import LatentDatasetAudio

class TestLatentDatasetAudio:
    @pytest.fixture
    def setup_test_data(self, tmp_path):
        # 创建临时测试数据目录结构
        dataset_dir = tmp_path / "test_dataset"
        os.makedirs(dataset_dir / "video", exist_ok=True)
        os.makedirs(dataset_dir / "latent", exist_ok=True)
        os.makedirs(dataset_dir / "prompt_embed", exist_ok=True)
        os.makedirs(dataset_dir / "prompt_attention_mask", exist_ok=True)
        os.makedirs(dataset_dir / "audio_emb", exist_ok=True)
        os.makedirs(dataset_dir / "face_emb", exist_ok=True)

        # 创建测试数据
        latent = torch.randn(1, 4, 32, 32)  # 示例潜在向量
        prompt_embed = torch.randn(256, 4096)  # 示例提示嵌入
        prompt_mask = torch.ones(256).bool()  # 示例提示掩码
        audio_emb = torch.randn(1000, 512)  # 示例音频嵌入
        face_emb = torch.randn(1000, 512)  # 示例人脸嵌入

        # 保存测试数据
        torch.save(latent, dataset_dir / "latent" / "test_latent.pt")
        torch.save(prompt_embed, dataset_dir / "prompt_embed" / "test_prompt.pt")
        torch.save(prompt_mask, dataset_dir / "prompt_attention_mask" / "test_mask.pt")
        torch.save(audio_emb, dataset_dir / "audio_emb" / "test_audio.pt")
        torch.save(face_emb, dataset_dir / "face_emb" / "test_face.pt")

        # 创建json注释文件
        json_data = [{
            "latent_path": "test_latent.pt",
            "prompt_embed_path": "test_prompt.pt",
            "prompt_attention_mask": "test_mask.pt",
            "audio_emb_path": "test_audio.pt",
            "face_emb_path": "test_face.pt",
            "length": 32
        }]
        
        json_path = dataset_dir / "test.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f)

        return str(json_path)

    def test_getitem_with_all_embeddings(self, setup_test_data):
        """测试当所有嵌入都存在时的__getitem__函数"""
        dataset = LatentDatasetAudio(
            json_path=setup_test_data,
            num_latent_t=28,
            cfg_rate=0.1
        )
        
        # 获取第一个数据项
        latent, prompt_embed, prompt_mask, audio_emb, face_emb = dataset[0]
        
        # 验证返回的张量形状
        assert isinstance(latent, torch.Tensor)
        assert isinstance(prompt_embed, torch.Tensor)
        assert isinstance(prompt_mask, torch.Tensor)
        assert isinstance(audio_emb, torch.Tensor)
        assert isinstance(face_emb, torch.Tensor)
        
        # 验证张量维度
        assert latent.shape[1] == 28  # 验证latent_t维度
        assert prompt_embed.shape == (256, 4096)
        assert prompt_mask.shape == (256,)
        assert audio_emb.shape[1] == 512
        assert face_emb.shape[1] == 512

    def test_getitem_with_cfg(self, setup_test_data):
        """测试cfg_rate=1.0时的__getitem__函数"""
        dataset = LatentDatasetAudio(
            json_path=setup_test_data,
            num_latent_t=28,
            cfg_rate=1.0  # 设置cfg_rate为1.0，确保使用uncond_prompt
        )
        
        latent, prompt_embed, prompt_mask, audio_emb, face_emb = dataset[0]
        
        # 验证是否使用了uncond_prompt
        assert torch.all(prompt_embed == dataset.uncond_prompt_embed)
        assert torch.all(prompt_mask == dataset.uncond_prompt_mask)

    def test_getitem_without_audio_face(self, setup_test_data, tmp_path):
        """测试没有音频和人脸嵌入时的__getitem__函数"""
        # 创建新的测试数据目录结构
        dataset_dir = tmp_path / "test_dataset_no_audio_face"
        os.makedirs(dataset_dir / "latent", exist_ok=True)
        os.makedirs(dataset_dir / "prompt_embed", exist_ok=True)
        os.makedirs(dataset_dir / "prompt_attention_mask", exist_ok=True)

        # 创建基本测试数据
        latent = torch.randn(1, 4, 32, 32)
        prompt_embed = torch.randn(256, 4096)
        prompt_mask = torch.ones(256).bool()

        # 保存测试数据
        torch.save(latent, dataset_dir / "latent" / "test_latent.pt")
        torch.save(prompt_embed, dataset_dir / "prompt_embed" / "test_prompt.pt")
        torch.save(prompt_mask, dataset_dir / "prompt_attention_mask" / "test_mask.pt")

        # 创建新的json数据，不包含音频和人脸嵌入
        json_data = [{
            "latent_path": "test_latent.pt",
            "prompt_embed_path": "test_prompt.pt",
            "prompt_attention_mask": "test_mask.pt"
        }]
        
        json_path = dataset_dir / "test.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f)
            
        dataset = LatentDatasetAudio(
            json_path=str(json_path),
            num_latent_t=28,
            cfg_rate=0.1
        )
        
        latent, prompt_embed, prompt_mask, audio_emb, face_emb = dataset[0]
        
        # 验证音频和人脸嵌入是否为None
        assert audio_emb is None
        assert face_emb is None

    def test_with_real_data(self):
        """使用实际数据测试数据集"""
        json_path = "/wangbenyou/shunian/workspace/talking_face/model_training/FastVideo/data/hallo3-data-origin-1k/videos2caption.json"
        
        # 确保文件存在
        assert os.path.exists(json_path), f"数据文件 {json_path} 不存在"
        
        dataset = LatentDatasetAudio(
            json_path=json_path,
            num_latent_t=32,  # 根据实际训练脚本中的参数设置
            cfg_rate=0.0  # 根据实际训练脚本中的参数设置
        )
        
        # 测试数据集长度
        assert len(dataset) > 0, "数据集为空"
        
        # 获取第一个数据项并验证
        try:
            latent, prompt_embed, prompt_mask, audio_emb, face_emb = dataset[0]
            
            # 验证返回的张量
            assert isinstance(latent, torch.Tensor), "latent 不是 tensor"
            assert isinstance(prompt_embed, torch.Tensor), "prompt_embed 不是 tensor"
            assert isinstance(prompt_mask, torch.Tensor), "prompt_mask 不是 tensor"
            
            # 验证基本维度
            # assert latent.shape[1] == 32, f"latent 维度错误: {latent.shape}"
            assert prompt_embed.shape == (256, 4096), f"prompt_embed 维度错误: {prompt_embed.shape}"
            assert prompt_mask.shape == (256,), f"prompt_mask 维度错误: {prompt_mask.shape}"
            
            # 如果有音频和人脸嵌入，验证它们
            if audio_emb is not None:
                assert isinstance(audio_emb, torch.Tensor), "audio_emb 不是 tensor"
                assert audio_emb.shape[1] == 512, f"audio_emb 维度错误: {audio_emb.shape}"
            
            if face_emb is not None:
                assert isinstance(face_emb, torch.Tensor), "face_emb 不是 tensor"
                assert face_emb.shape[1] == 512, f"face_emb 维度错误: {face_emb.shape}"
                
            print(f"数据集大小: {len(dataset)}")
            print(f"latent shape: {latent.shape}")
            print(f"prompt_embed shape: {prompt_embed.shape}")
            print(f"prompt_mask shape: {prompt_mask.shape}")
            print(f"audio_emb shape: {audio_emb.shape if audio_emb is not None else None}")
            print(f"face_emb shape: {face_emb.shape if face_emb is not None else None}")
            
        except Exception as e:
            pytest.fail(f"加载数据时出错: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__]) 