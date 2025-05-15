import numpy as np
import pandas as pd
import librosa
import torch
import laion_clap
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import pickle as pk

ext = ".wav"

gen_dir = "/data2/klu15/audioldm_eval/audio/generate"
ref_dir = "/data2/klu15/audioldm_eval/audio/reference"

csv_dir = "/data2/klu15/audioldm_eval/map_csv"

datasets = [
    "music_caps",
    # "song_describer",
    # "smm",
]

models = [
    # "musiclm", 
    # "stable_audio",
    # "musicgen", 
    # "audioldm2", 
    # "mustango", 
    "mureka",
]

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)


def clap_score(model):
    score_dic = {dataset : {gen_model : {} for gen_model in models} for dataset in datasets}
    clap_dic = {dataset : {gen_model : {} for gen_model in models} for dataset in datasets}
    for dataset in datasets:
        df = pd.read_csv(f"{csv_dir}/{dataset}.csv")
        for gen_model in models:
            for idx in tqdm(df.index, unit = "row", desc = f"{dataset}|{gen_model}"):
                caption = (df.iloc[idx]["prompt"] if dataset == "smm" else df.iloc[idx]["caption"])
                fp = f"{gen_dir}/{dataset}/{gen_model}/{idx}{ext}"
                if not os.path.isfile(fp):
                    continue
                with torch.no_grad():
                    caption_embeds = model.get_text_embedding([caption], use_tensor=True)
                    audio_data, _ = librosa.load(fp, sr=48000) 
                    audio_data = audio_data.reshape(1, -1) 
                    audio_data = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float()
                    audio_embeds = model.get_audio_embedding_from_data(x = audio_data, use_tensor=True)
                    cos_sim_torch = F.cosine_similarity(audio_embeds, caption_embeds, dim=-1)
                    # cos_sim.append(cos_sim_torch)
                    clap_dic[dataset][gen_model].update({idx : cos_sim_torch})
            clap_dic_values = torch.Tensor(list(clap_dic[dataset][gen_model].values()))
            clap_mean = torch.mean(clap_dic_values)
            clap_std = torch.std(clap_dic_values)
            print(f"{dataset}|{gen_model}, CS_mean : {clap_mean}, CS_std : {clap_std}")
            score_dic[dataset][gen_model].update({"mean" : clap_mean, "std" : clap_std})
    return score_dic

if __name__ == '__main__':
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    scores = clap_score(model)
    print(scores)
    
