import os
import glob
import torch
import pandas as pd
import csv
from .audioldm_eval import EvaluationHelper, EvaluationHelperParallel
import torch.multiprocessing as mp
import ipdb

device = torch.device(f"cuda:{0}")

generation_result_path = "/data2/klu15/audioldm_eval/audio/generate"
target_audio_path = "/data2/klu15/audioldm_eval/audio/reference"

audio_paths = [
    "musiclm", 
    "stable_audio", 
    "musicgen", 
    "audioldm2",
    "mustango", 
    "mureka"
]

datasets = [
    "music_caps", 
    "song_describer", 
    "smm"
]

def run_ldm_eval(args):
    import audioldm_eval
    print(audioldm_eval.__file__)


    generate = f"{args.gen_dir}"
    reference = f"{args.ref_dir}"

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} CUDA device(s)")
    
    if num_gpus > 1:
        evaluator = EvaluationHelperParallel(16000, num_gpus=num_gpus)
    else:
        evaluator = EvaluationHelper(16000, device)
        
    metrics = evaluator.main(
        generate,
        reference,
    )

if __name__ == '__main__':
    
    evaluator = EvaluationHelperParallel(16000, 4) 
    
    for dataset in datasets:
        for folder in audio_paths:
            generate = f"{generation_result_path}/{dataset}/{folder}"
            reference = f"{target_audio_path}/{dataset}"
            metrics = evaluator.main(
                generate,
                reference,
            )
    