import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)))
import librosa
import subprocess
import torch
import numpy as np
import laion_clap
from typing import Union, List
from base_encoder import BaseEncoder
from laion_clap.training.data import int16_to_float32, float32_to_int16, get_audio_features

class CLAPEncoder(BaseEncoder):
    """CLAP encoder implementation supporting both audio and text modalities"""
    
    def __init__(self, model_path: str = None, device: str = 'cpu'):
        super().__init__(model_path, device)
        self.model = None
        self.model_path = model_path if model_path else "./models/CLAP/ckpt/music_audioset_epoch_15_esc_90.14.pt"
        self.sample_rate = 48000
        
    def load_model(self):
        """加载CLAP模型权重和预处理工具"""
        if not self.initialized:
            self.model = laion_clap.CLAP_Module(
                enable_fusion=False,
                amodel='HTSAT-base'
            ).to(torch.device(self.device))
            
            if self.model_path:
                self.model.load_ckpt(self.model_path)
                
            self.model.eval()                
            self.initialized = True

    def get_modality_support(self):
        """返回支持的模态类型"""
        return {'audio': True, 'text': True}

    @BaseEncoder._batch_processing_wrapper
    def encode_audio(self, 
                    audio_input: Union[str, np.ndarray, torch.Tensor, List[str], bytes],
                    **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
        """统一音频编码入口，支持多种输入格式"""
        self._check_initialized()
        
        # 转换输入为文件路径格式（处理字节流等复杂情况）
        audio_path = self._handle_audio_input(audio_input)
        
        # 使用优化后的处理流程
        audio_embed = self.model.get_audio_embedding_from_filelist(x = [audio_path], use_tensor=True)[0]
        return audio_embed.cpu().detach().numpy()

    def encode_text(self,
                   text_input: Union[str, List[str], dict],
                   **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
        """文本编码实现"""
        self._check_initialized()
        
        if isinstance(text_input, dict):
            texts = text_input['text']
        elif isinstance(text_input, str):
            texts = [text_input]
        else:
            texts = text_input
            
        with torch.no_grad():
            text_embeddings = self.model.get_text_embedding(texts)
            
        return torch.stack(text_embeddings) if len(texts) > 1 else text_embeddings[0]

    def _handle_audio_input(self, audio_input):
        """处理不同类型的音频输入"""
        # 如果是字节流，先保存为临时文件
        if isinstance(audio_input, bytes):
            return self._bytes_to_tempfile(audio_input)
            
        # 如果是张量或数组，转换为文件保存
        if isinstance(audio_input, (np.ndarray, torch.Tensor)):
            return self._array_to_tempfile(audio_input)
            
        return audio_input

    def _load_and_process_audio(self, file_path: str) -> torch.Tensor:
        """加载并处理音频波形"""
        waveform, _ = librosa.load(file_path, sr=self.sample_rate)
        waveform = self.int16_to_float32(self.float32_to_int16(waveform))
        return torch.from_numpy(waveform).float().to(self.device)

    def int16_to_float32(self, x):
        return (x / 32767.0).astype(np.float32)

    def float32_to_int16(self, x):
        x = np.clip(x, a_min=-1., a_max=1.)
        return (x * 32767.).astype(np.int16)
