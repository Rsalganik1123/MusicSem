import pandas as pd 
from glob import glob 
import os 
import re 
import shutil
import argparse  
import ipdb 

from eval_clap import run_clap_eval 
from eval_ldm import run_ldm_eval 
from eval_vendi import run_vendi_eval 

prompt_dict = {
    'music_caps': '/data2/rsalgani/music-caps/musiccaps-public.csv', 
    'song_describer': '/data2/rsalgani/song_describer/song_describer.csv',
    'test_set_May3': '/data2/rsalgani/hallucination/test.csv', 
    'test_set_May6': '/data2/rsalgani/reddit/test_sets/test_set_May6/package/test_May6_final.csv'
}

audio_dict = {
    'music_caps': ['/data2/rsalgani/music-caps/wav/', 'wav'], 
    'song_describer': ['/data2/rsalgani/song_describer/audio/', 'mp3'],
    'test_set_May6': []
}

def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_model', choices=['musicgen', 'stable_audio', 'musiclm', 'mustango', 'audioldm2', 'mureka'])
    parser.add_argument('--gen_dir', default='/data2/rsalgani/saved_embeddings/')
    parser.add_argument('--ref_dir' default='')
    parser.add_argument('--dataset', choices=['music_caps', 'song_describer', 'test_set_May6'])
    parser.add_argument('--metric', choices=['CLAP', 'KLD', 'VS'])
    return parser.parse_args()

    
if __name__ == "__main__": 
    args = parse_args() 
    data_path = audio_dict[args.dataset]
    if args.metric == 'CLAP': 
        run_clap_eval(args, data_path)
