import numpy as np
import torch
import librosa
import subprocess
import os
import laion_clap

from laion_clap.training.data import int16_to_float32, float32_to_int16, get_audio_features
import torch.nn.functional as F

import torch

class CLAP():
    def __init__(self, ckpt_path="./models/CLAP/ckpt/clap/music_audioset_epoch_15_esc_90.14.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base').to(self.device)
        self.model.load_ckpt(ckpt_path)

    def get_audio_embedding(self, audio_path):
        temp_audio_path = f"{audio_path}_temp.wav"
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if librosa.get_duration(filename=audio_path) > 60:
            command = [
                'ffmpeg', '-i', audio_path,
                '-ss', '0', '-to', '60',
                '-ar', '24000', '-ab', '24k',
                temp_audio_path
            ]
        else:
            command = [
                'ffmpeg', '-i', audio_path,
                '-ar', '24000', '-ab', '24k',
                temp_audio_path
            ]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        audio_embed = self.model.get_audio_embedding_from_filelist(x=[temp_audio_path], use_tensor=True)
        os.remove(temp_audio_path)
        
        return audio_embed[0].detach().clone()
    
    def get_audio_embedding_v2(self, audio_path):
        temp_audio_path = f"{audio_path}_temp.wav"
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if librosa.get_duration(filename=audio_path) > 60:
            command = [
                'ffmpeg', '-i', audio_path,
                '-ss', '0', '-to', '60',
                '-ar', '48000', '-ab', '24k',
                temp_audio_path
            ]
        else:
            command = [
                'ffmpeg', '-i', audio_path,
                '-ar', '48000', '-ab', '24k',
                temp_audio_path
            ]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        x = [temp_audio_path]
        
        self.model.eval()
        audio_input = []
        for f in x:
            # load the waveform of the shape (T,), should resample to 48000
            audio_waveform, _ = librosa.load(f, sr=48000)           
            # quantize
            audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
            audio_waveform = torch.from_numpy(audio_waveform).float()
            temp_dict = {}
            temp_dict = get_audio_features(
                temp_dict, audio_waveform, 480000, 
                data_truncating='fusion', 
                data_filling='repeatpad',
                audio_cfg=self.model.model_cfg['audio_cfg'],
                require_grad=audio_waveform.requires_grad
            )
            audio_input.append(temp_dict)
            
        data = audio_input
        device = next(self.model.parameters()).device
        input_dict = {}
        keys = data[0].keys()
        for k in keys:
            input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in data], dim=0).to(device)
        audio_embeds = self.model.model.encode_audio(input_dict, device=device)["fine_grained_embedding"] # modifed original code: /home/tteng/miniconda3/envs/mlm/lib/python3.10/site-packages/laion_clap/clap_module/htsat.py, line 767
        # import pdb;pdb.set_trace()
        os.remove(temp_audio_path)
        
        return audio_embeds[0].detach().clone()


