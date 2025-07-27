import os
import json
import torch
import librosa
import numpy as np
from typing import Union, List, Dict
# import ruamel.yaml as yaml
from glob import glob
from tqdm import tqdm
from ruamel.yaml import YAML

import warnings

# Ignore all warnings
warnings.simplefilter("ignore")

from LARP.models.larp import LARP

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)))
from base_encoder import BaseEncoder

class LARPEncoder(BaseEncoder):
    """LARP multimodal encoder implementation supporting fusion of audio and text"""
    
    def __init__(self, model_path: str = None, device: str = 'cpu', metadata_path: str = None):
        super().__init__(model_path, device)
        self.model = None
        self.metadata = None
        self.audio_config = None
        self.text_config = None
        self._default_config = "config.yaml"
        self._default_checkpoint = "models/LARP/LFM/TPC.pth"
        self.model_path = "/model/tteng/MusicEnocdeFactory/models/LARP"
        
        # Initialize metadata
        if metadata_path:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)

    def load_model(self):
        if not self.initialized:
            config_path = os.path.join(self.model_path, self._default_config)
            yaml = YAML(typ='safe', pure=True)

            with open(config_path, 'r') as f:
                config = yaml.load(f)
            
            base_config = config['MODELS']['BASE_MODELS']['LARP']
            base_config.update({
                'device': self.device,
                'num_tracks': 87558,  # Adjust based on actual dataset
                'num_playlists': 1
            })
            
            self.audio_config = config['MODELS']['AUDIO_MODELS']['HTSAT']
            self.text_config = config['MODELS']['LANGUAGE_MODELS']['BERT']
            
            self.model = LARP(
                config=base_config,
                audio_cfg=self.audio_config,
                text_cfg=self.text_config,
                embed_dim=base_config['embed_dim']
            ).to(torch.device(self.device))
            
            checkpoint_path = os.path.join(self.model_path, self._default_checkpoint)
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            state_dict = checkpoint['model']
            self.model.load_state_dict(state_dict, strict=False)
            
            self.model.eval()
            self.initialized = True

    def get_modality_support(self):
        """Return supported modality types"""
        return {'audio': True, 'text': True}

    @BaseEncoder._batch_processing_wrapper
    def encode_audio(self, 
                    audio_input: Union[str, np.ndarray, torch.Tensor, List[str], bytes],
                    **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Encode audio input (supports 5 input formats)"""
        self._check_initialized()
        
        processed_path = self._handle_audio_input(audio_input)
        
        audio = self._load_audio(processed_path)       
        
        with torch.no_grad(): 
            audio_embed = self.model.return_features(audio, None)[0]
        
        return self._post_process(audio_embed)


    @BaseEncoder._batch_processing_wrapper
    def encode_text(self,
                   text_input: Union[str, List[str], dict],
                   **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
        self._check_initialized()
        
        if isinstance(text_input, dict):
            texts = text_input['text']
        elif isinstance(text_input, str):
            texts = [text_input]
        else:
            texts = text_input
            
        with torch.no_grad():
            text_embeddings = self.model.return_features(None,texts)[1]
            
        return self._post_process(text_embeddings)

    def _handle_audio_input(self, data):
        """Unified processing for different types of audio input"""
        # Handle byte streams
        if isinstance(data, bytes):
            return self._save_bytes(data)
            
        # Handle arrays/tensors
        if isinstance(data, (np.ndarray, torch.Tensor)):
            return self._save_array(data)
            
        return data  # Return file path directly

    def _load_audio(self, path: str) -> torch.Tensor:
        """Load and preprocess audio"""
        audio, sr = librosa.load(path, sr=self.audio_config.get('sample_rate', 22050))
        audio = librosa.util.normalize(audio) * 0.95
        return torch.from_numpy(audio).float().unsqueeze(0)

    def _save_bytes(self, data: bytes) -> str:
        """Save byte stream to temporary file"""
        temp_path = "/tmp/larp_audio_temp.wav"
        with open(temp_path, 'wb') as f:
            f.write(data)
        return temp_path

    def _save_array(self, data: Union[np.ndarray, torch.Tensor]) -> str:
        """Save array to temporary file"""
        temp_path = "/tmp/larp_array_temp.wav"
        data = data.cpu().numpy() if isinstance(data, torch.Tensor) else data
        librosa.output.write_wav(temp_path, data, self.audio_config.get('sample_rate', 22050))
        return temp_path

    def _cleanup_temp_file(self, path: str):
        """Clean up temporary audio files"""
        if path.startswith('/tmp/larp_') and os.path.exists(path):
            try:
                os.remove(path)
            except:
                pass

    def _check_initialized(self):
        if not self.initialized:
            raise RuntimeError("Encoder not initialized. Call load_model() first")
        
    def _post_process(self, embeddings: torch.Tensor):
        """Post-processing: device conversion and dimension reduction"""
        embeddings = embeddings.cpu() if self.device == 'cpu' else embeddings
        return embeddings.squeeze(0).cpu().detach().numpy() if embeddings.dim() > 1 else embeddings