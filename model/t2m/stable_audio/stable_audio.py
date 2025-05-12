import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
import os 
from tqdm import tqdm 
import pandas as pd 
import pickle 
import warnings
warnings.filterwarnings("ignore")
os.environ['HF_HOME'] = '/data2/rsalgani/hub/'

def stable_audio_gen(args, dataset_dict): 
    device = args.device 

    # Download model
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    model = model.to(device)
    data_path = dataset_dict[args.dataset]

    if 'pkl' in data_path: 
        df = pickle.load(open(data_path, 'rb'))
    elif 'csv' in data_path: 
        df = pd.read_csv(data_path)
    else: 
        raise NotImplemented 
    
    for i in tqdm(range(len((df)))): 
        row = df.iloc[i, :]
        idx = row.idx 
        cap = row[args.prompt_key]
            
            
        conditioning = [{
            "prompt": f"{cap}",
            "seconds_start": 0, 
            "seconds_total": 30
        }]

        # Generate stereo audio
        output = generate_diffusion_cond(
            model,
            steps=100,
            cfg_scale=7,
            conditioning=conditioning,
            sample_size=sample_size,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-3m-sde",
            device=device
        )

        # Rearrange audio batch to a single sequence
        output = rearrange(output, "b d n -> d (b n)")

        # Peak normalize, clip, convert to int16, and save to file
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        torchaudio.save(f"{args.save_dir}{args.dataset}/stable_audio/{i}.wav", output, sample_rate)