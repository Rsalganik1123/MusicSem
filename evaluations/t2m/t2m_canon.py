import pandas as pd 
from glob import glob 
import os 
import re 
import shutil
import argparse  
import ipdb 



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
    parser.add_argument('--mode', choices=['eval', 'gather'])
    parser.add_argument('--gen_model', choices=['musicgen', 'stable_audio', 'musiclm', 'mustango', 'audioldm2'])
    parser.add_argument('--mc_path', default='/data2/rsalgani/music-caps/wav/')
    parser.add_argument('--audio_save_dir', default='/data2/rsalgani/saved_embeddings/')
    parser.add_argument('--save_dir', default='/data2/rsalgani/eval_outputs/fad/')
    return parser.parse_args()

