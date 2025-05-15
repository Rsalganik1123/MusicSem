import numpy as np
from typing import Union, List
import torch
import librosa
import os
import subprocess

class BaseEncoder():
    """Abstract base class for music-text encoders"""
    
    def __init__(self, model_path: str = None, device: str = 'cpu'):
        """Initialize encoder
        Args:
            model_path: Path to model weights/configs
            device: Target device (cpu/cuda)
        """
        self.model_path = model_path
        self.device = device
        self.initialized = False
    
    def load_model(self):
        """Load model weights and preprocessing tools"""
        pass

    def _check_initialized(self):
        if not self.initialized:
            raise RuntimeError("Encoder not initialized. Call load_model() first.")
    
    def get_modality_support(self):
        """Return supported modalities
        Returns:
            dict: {'audio': bool, 'text': bool}
        """
        pass
    
    def encode_audio(self, 
                    audio_input: Union[str, np.ndarray, torch.Tensor, List[str], bytes],
                    **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Encode audio input with flexible format support"""
        raise NotImplementedError("encode_audio() not implemented for this encoder")

    def encode_text(self,
                   text_input: Union[str, List[str], dict],
                   **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Encode text input with flexible format support"""
        raise NotImplementedError("encode_text() not implemented for this encoder")
    
    def __call__(self, modality: str, input_data):
        """Unified call interface"""
        if modality == 'audio':
            return self.encode_audio(input_data)
        elif modality == 'text':
            return self.encode_text(input_data)
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        
    @staticmethod
    def _convert_audio_input(input_data, target_type=np.ndarray):
        """Helper method for common audio input conversions"""
        if isinstance(input_data, target_type):
            return input_data
        
        raise ValueError(f"Cannot convert {type(input_data)} to {target_type}")

    @staticmethod
    def _batch_processing_wrapper(func):
        """Decorator for handling batch inputs"""
        def wrapper(self, input_data, *args, **kwargs):
            if isinstance(input_data, list):
                return [func(self, item, *args, **kwargs) for item in input_data]
            return func(self, input_data, *args, **kwargs)
        return wrapper
    
    def _preprocess_audio(self, audio_path: str) -> str:
        temp_path = f"{audio_path}_clap_temp.wav"
        
        if librosa.get_duration(path=audio_path) > 60:
            cmd = ['ffmpeg', '-i', audio_path, '-ss', '0', '-to', '60',
                   '-ar', str(self.sample_rate), '-ab', '24k', temp_path]
        else:
            cmd = ['ffmpeg', '-i', audio_path, 
                   '-ar', str(self.sample_rate), temp_path]
                   
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return temp_path
    
    def _bytes_to_tempfile(self, data: bytes) -> str:
        temp_path = "/tmp/clap_temp_{}.wav".format(hash(data))
        with open(temp_path, 'wb') as f:
            f.write(data)
        return temp_path

    def _array_to_tempfile(self, array: Union[np.ndarray, torch.Tensor]) -> str:
        temp_path = "/tmp/clap_array_temp.wav"
        array = array.cpu().numpy() if isinstance(array, torch.Tensor) else array
        librosa.output.write_wav(temp_path, array, self.sample_rate)
        return temp_path

    def _cleanup_tempfile(self, path: str):
        if os.path.exists(path):
            try:
                os.remove(path)
            except:
                pass