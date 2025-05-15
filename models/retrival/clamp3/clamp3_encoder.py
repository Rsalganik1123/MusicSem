import os
import sys
import shutil
import tempfile
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import Resample
from typing import Union, List
import os
import torch
import numpy as np
from transformers import BertConfig, AutoTokenizer

from .code.config import *
from .code.utils import *
from .preprocessing.audio.hf_pretrains import HuBERTFeature
from .preprocessing.audio.extract_mert import mert_infr_features
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)))
from base_encoder import BaseEncoder



class CLAMP3Encoder(BaseEncoder):    
    def __init__(self, 
                 model_path: str = "/model/tteng/MusicEnocdeFactory/models/clamp3/code/weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth",
                 device: str = 'cuda',
                 text_model_name: str = TEXT_MODEL_NAME,
                 max_length: int = 128):
        super().__init__(model_path, device)
        
        self.target_sr = 24000 
        self.resampler = None 
        
        self.text_model_name = text_model_name
        self.max_length = max_length
        
        self.tokenizer = None
        self.model = None
        self.hubert = None

    def load_model(self):
        if self.initialized:
            return
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        
        symbolic_config = self._create_symbolic_config()
        audio_config = self._create_audio_config()
        
        self.model = CLaMP3Model(  
            audio_config=audio_config,
            symbolic_config=symbolic_config,
            text_model_name=self.text_model_name,
            hidden_size=768,
            load_m3=False
        ).to(self.device)
        
        self._load_pretrained_weights()
        
        self.hubert = HuBERTFeature(
        "m-a-p/MERT-v1-95M",
        24000,
        force_half=False,
        processor_normalize=True,
        )
        self.hubert.to(self.device)
        self.hubert.eval()
            
        self.model.eval()
        self.initialized = True

    def _create_symbolic_config(self):
        return BertConfig(vocab_size=1,
                            hidden_size=M3_HIDDEN_SIZE,
                            num_hidden_layers=PATCH_NUM_LAYERS,
                            num_attention_heads=M3_HIDDEN_SIZE//64,
                            intermediate_size=M3_HIDDEN_SIZE*4,
                            max_position_embeddings=PATCH_LENGTH)

    def _create_audio_config(self):
        return BertConfig(vocab_size=1,
                        hidden_size=AUDIO_HIDDEN_SIZE,
                        num_hidden_layers=AUDIO_NUM_LAYERS,
                        num_attention_heads=AUDIO_HIDDEN_SIZE//64,
                        intermediate_size=AUDIO_HIDDEN_SIZE*4,
                        max_position_embeddings=MAX_AUDIO_LENGTH)


    def _load_pretrained_weights(self):
        if not self.model_path:
            raise ValueError("Model path must be specified")
            
        checkpoint = torch.load(self.model_path, map_location="cpu")
        self.model.load_state_dict(checkpoint['model'])

    def get_modality_support(self):
        return {'audio': True, 'text': True}

    def encode_text(self, text_input: str, **kwargs) -> torch.Tensor:
        self._check_initialized()
        
        # 文件读取逻辑
        if os.path.isfile(text_input):
            with open(text_input, 'r', encoding='utf-8') as f:  # 修复编码问题
                text = '\n'.join(list(set(f.read().splitlines())))
        else:
            text = text_input
        input_data = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_data = input_data['input_ids'].squeeze(0)
        max_input_length = MAX_TEXT_LENGTH
            
        with torch.no_grad():
            return self._encoder_core(input_data, max_input_length, modality="text").cpu().detach().numpy()
        
    def encode_audio(self, audio_input: Union[str, np.ndarray], **kwargs) -> torch.Tensor:
        self._check_initialized()
        
        with torch.no_grad():
            wav = mert_infr_features(self.hubert, audio_input, self.device)
            # print(wav.shape)
            wav = wav.mean(dim=0, keepdim=True)
            # print(wav.shape)
            input_data = wav.reshape(-1, wav.size(-1)).to(self.device)
            zero_vec = torch.zeros((1, input_data.size(-1))).to(self.device)
            input_data = torch.cat((zero_vec, input_data, zero_vec), 0)
            max_input_length = MAX_AUDIO_LENGTH
            return self._encoder_core(input_data, max_input_length, modality="audio").cpu().detach().numpy()

    def __call__(self, modality: str, input_data):
        if modality not in ['text', 'audio']:
            raise ValueError(f"Unsupported modality: {modality}")
        return super().__call__(modality, input_data)

    def _encoder_core(self, input_data, max_input_length, modality="text"):
        segment_list = []
        for i in range(0, len(input_data), max_input_length):
            segment_list.append(input_data[i:i+max_input_length])
        segment_list[-1] = input_data[-max_input_length:]

        last_hidden_states_list = []

        for input_segment in segment_list:
            input_masks = torch.tensor([1]*input_segment.size(0))
            if modality=="text":
                pad_indices = torch.ones(MAX_TEXT_LENGTH - input_segment.size(0)).long() * self.tokenizer.pad_token_id
            else:
                pad_indices = torch.ones((MAX_AUDIO_LENGTH - input_segment.size(0), AUDIO_HIDDEN_SIZE)).float() * 0.
            pad_indices = pad_indices.to(self.device)
            input_masks = torch.cat((input_masks, torch.zeros(max_input_length - input_segment.size(0))), 0).to(self.device)
            # print(input_segment.shape)
            # print(pad_indices.shape)
            input_segment = torch.cat((input_segment, pad_indices), 0).to(self.device)

            if modality=="text":
                last_hidden_states = self.model.get_text_features(text_inputs=input_segment.unsqueeze(0).to(self.device),
                                                            text_masks=input_masks.unsqueeze(0).to(self.device),
                                                            get_global=True)
            else:
                last_hidden_states = self.model.get_audio_features(audio_inputs=input_segment.unsqueeze(0).to(self.device),
                                                            audio_masks=input_masks.unsqueeze(0).to(self.device),
                                                            get_global=True)
            last_hidden_states_list.append(last_hidden_states)

        full_chunk_cnt = len(input_data) // max_input_length
        remain_chunk_len = len(input_data) % max_input_length
        if remain_chunk_len == 0:
            feature_weights = torch.tensor([max_input_length] * full_chunk_cnt, device=self.device).view(-1, 1)
        else:
            feature_weights = torch.tensor([max_input_length] * full_chunk_cnt + [remain_chunk_len], device=self.device).view(-1, 1)
        
        last_hidden_states_list = torch.concat(last_hidden_states_list, 0)
        last_hidden_states_list = last_hidden_states_list * feature_weights
        last_hidden_states_list = last_hidden_states_list.sum(dim=0) / feature_weights.sum()
        return last_hidden_states_list
