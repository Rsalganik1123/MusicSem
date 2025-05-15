import os
import glob
import torch
import pandas as pd
import csv
from audioldm_eval import EvaluationHelper, EvaluationHelperParallel
import torch.multiprocessing as mp
device = torch.device(f"cuda:{0}")
import ipdb

generation_result_path = "/data2/klu15/audioldm_eval/audio/generate"
target_audio_path = "/data2/klu15/audioldm_eval/audio/reference"

audio_paths = [
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
    # "smm"
]

## Multiple GPUs

if __name__ == '__main__':
    
    # rename_musiccaps_ref(mc_csv)
    # rename_musiccaps_ref(sd_csv)
    
    evaluator = EvaluationHelperParallel(16000, 8) 
    
    for dataset in datasets:
        for folder in audio_paths:
            generate = f"{generation_result_path}/{dataset}/{folder}"
            reference = f"{target_audio_path}/{dataset}"
            metrics = evaluator.main(
                generate,
                reference,
            )
            ipdb.set_trace()
    