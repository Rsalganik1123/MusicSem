import os
import torch
import tempfile
import numpy as np
from typing import Union, List, Dict
# sys.path.append(os.path.abspath)
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)))
from base_encoder import BaseEncoder

class ImageBindEncoder(BaseEncoder):
    """ImageBind multimodal encoder implementation supporting text and audio modalities"""
    
    def __init__(self, model_path: str = None, device: str = 'cpu'):
        super().__init__(model_path, device)
        self.model = None
        self.temp_dir = None
        self.model_ckpt_path = "/model/tteng/MusicEnocdeFactory/models/imagebind/ckpt/imagebind_huge.pth"
        if os.path.exists(self.model_ckpt_path):
            self.model = imagebind_model.imagebind_huge(pretrained=False)
            self.model.load_state_dict(torch.load(self.model_ckpt_path, map_location=self.device))
        else:
            print(f"Model checkpoint not found at {self.model_ckpt_path}, downloading pretrained model...")
            self.model = imagebind_model.imagebind_huge(pretrained=True)
            os.makedirs('./model', exist_ok=True)
            torch.save(self.model.state_dict(), self.model_ckpt_path)
        self.initialized = True
        self.model.eval()
        self.model.to(self.device)
    
    def get_modality_support(self):
        """Return supported modality types"""
        return {'audio': True, 'text': True}

    @BaseEncoder._batch_processing_wrapper
    def encode_audio(self, 
                    audio_input: Union[str, np.ndarray, torch.Tensor, List[str], bytes],
                    **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Encode audio input (supports 5 input formats)"""
        self._check_initialized()
        
        # Unified input processing
        processed_paths = self._process_audio_input(audio_input)
        
        # Load and transform audio data
        inputs = {
            ModalityType.AUDIO: data.load_and_transform_audio_data(
                processed_paths, 
                device=self.device
            )
        }
        
        # Perform inference
        with torch.no_grad():
            embeddings = self.model(inputs)[ModalityType.AUDIO]
        
        # Device consistency handling
        return self._post_process(embeddings).squeeze(0).cpu().numpy()

    @BaseEncoder._batch_processing_wrapper
    def encode_text(self,
                   text_input: Union[str, List[str], dict],
                   **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Encode text input (supports 3 input formats)"""
        self._check_initialized()
        
        # Unify input format
        texts = self._process_text_input(text_input)
        
        # Load and transform text data
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(
                texts, 
                device=self.device
            )
        }
        
        # Perform inference
        with torch.no_grad():
            embeddings = self.model(inputs)[ModalityType.TEXT]
        
        return self._post_process(embeddings).squeeze(0).cpu().numpy()

    def _process_audio_input(self, audio_input):
        """Unified processing for different types of audio input"""
        if isinstance(audio_input, list):
            return [self._save_non_file_input(f) if not isinstance(f, str) else f for f in audio_input]
        
        if isinstance(audio_input, (bytes, np.ndarray, torch.Tensor)):
            return [self._save_non_file_input(audio_input)]
        
        return [audio_input]

    def _process_text_input(self, text_input):
        """Unified processing for different types of text input"""
        if isinstance(text_input, dict):
            return text_input.get('text', [])
        elif isinstance(text_input, str):
            return [text_input]
        return text_input

    def _save_non_file_input(self, data: Union[bytes, np.ndarray, torch.Tensor]) -> str:
        """Save non-file input to temporary file"""
        self._setup_temp_dir()
        temp_path = os.path.join(self.temp_dir.name, f"temp_{hash(str(data))}.wav")
        
        if isinstance(data, bytes):
            with open(temp_path, 'wb') as f:
                f.write(data)
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            data = data.cpu().numpy() if isinstance(data, torch.Tensor) else data
            data.tofile(temp_path)
        
        return temp_path

    def _post_process(self, embeddings: torch.Tensor):
        """Post-processing: device conversion and dimension reduction"""
        embeddings = embeddings.cpu() if self.device == 'cpu' else embeddings
        return embeddings.squeeze(0) if embeddings.dim() > 2 else embeddings