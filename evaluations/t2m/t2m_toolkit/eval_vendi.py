'''
Log Apr.3,2025 - Kaifeng Lu

Problem:  CLAP has stopped maintaining dependencies since 2 years ago. Try to create a environment that runs CLAP seperately:

    ```
    conda create env -n "clap" python=3.10 numpy=1.24
    conda activate clap
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r requirements.txt
    ```

'''
import numpy as np
import librosa
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import laion_clap
import os
from .vendi import compute_vendi_score
import pickle as pk
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

audio_path = "/data2/klu15/audioldm_eval/audio/generate"

models = [
    # "musiclm", 
    # "stable_audio", 
    # "musicgen", 
    # "audioldm2",
    # "mustango", 
    "mureka"
]

datasets = [
    "music_caps", 
    # "song_describer", 
    # "smm",
]

checkpoint = None

# Load model checkpoint
def load_CLAP_ckpt(checkpoint=checkpoint):
    model = laion_clap.CLAP_Module(enable_fusion=False)
    if checkpoint:
        model.load_ckpt(checkpoint) # Load ckpt
    else:
        model.load_ckpt() # Load the best ckpt from fugging face automatically
    model.eval()
    return model

def power_to_db(P, ref_value=None, amin=1e-10, top_db=None):
    """
    Torch implementation of librosa.power_to_db(S_power, ref=np.max).
    """
    if ref_value is None:
        ref_value = P.max()
    log_spec = 10.0 * torch.log10(torch.clamp(P, min=amin))
    log_spec -= 10.0 * torch.log10(torch.tensor(ref_value))
    if top_db is not None:
        log_spec = torch.clamp(log_spec, max=log_spec.max() - top_db)
    return log_spec

# Generate spectrograms using librosa
def draw_spectrograms(audio_dir):
    specs = []
    print('Drawing spectrograms... ')
    for root, dirs, files in os.walk(audio_dir):
        for file in tqdm(files, unit="file"):
            if file.lower().endswith(".wav"):   # catches .wav, .WAV, etc.
                file_path = os.path.join(root, file)
                y, sr = librosa.load(file_path)
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                S_dB = librosa.power_to_db(S, ref=np.max)
                specs.append(S_dB)
    print('Finish drawing spectrograms.')
    return specs
    
# Evaluate against the metrics
def compute_vendi_from_spectrograms(specs: list[np.ndarray]) -> float:
    
    t_max = max(s.shape[1] for s in specs)

    padded = []
    print("Computing vendi scores...")
    for s in tqdm(specs):
        pad_amt = t_max - s.shape[1]
        if pad_amt > 0:
            pad_vals = s.min()
            s_p = np.pad(s,
                         pad_width=((0, 0), (0, pad_amt)),
                         mode='constant',
                         constant_values=pad_vals)
        else:
            s_p = s[:, :t_max]
        padded.append(s_p)

    flat = [s_p.flatten() for s_p in padded]
    X = np.stack(flat, axis=0).astype(np.float32) 

    return compute_vendi_score(X)

def run():
    df = pd.DataFrame(
        np.zeros((len(datasets), len(models))),
        columns=models,
        index=datasets,
    )
    for dataset in datasets:
        for model in models:
            audio_dir = f'{audio_path}/{dataset}/{model}'
            print(f"Computing Vendi score for /{dataset}/{model}")
            specs = draw_spectrograms(audio_dir)
            score = compute_vendi_from_spectrograms(specs)
            print(f"Vendi for /{dataset}/{model}: {score}")
            df.at[dataset, model] = score
    print(df)
    df.to_pickle("./vendi_score.pkl")

def run_vendi_eval(args):
    audio_dir = args.gen_dir
    print(f"Computing Vendi score for audio files: {audio_dir}")
    specs = draw_spectrograms(audio_dir)
    score = compute_vendi_from_spectrograms(specs)
    print("Vendi score: ",score)

if __name__ == "__main__": 
    run()