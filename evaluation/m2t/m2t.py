import pandas as pd 
from glob import glob 
import os 
import re 
import shutil
import argparse  
import ipdb 
from playntell.playntell.eval import run_eval 

def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data', choices=['song_describer', 'music_caps'])
    parser.add_argument('--gen_model', choices=['lp_musiccaps', 'futga', 'mullama'])
    parser.add_argument('--track_id', choices=['ytid', 'track_id'])
    parser.add_argument('--table_caption_key', choices=['caption', 'prompt'], default='caption')
    parser.add_argument('--gen_caption_key', default='captions_as_str')
    parser.add_argument('--csv_path')
    return parser.parse_args()

prompt_dict = {
    'music_caps': '/data2/rsalgani/music-caps/musiccaps-public.csv', 
    'song_describer': '/data2/rsalgani/song_describer/song_describer.csv',
    'test_set_May3': '/data2/rsalgani/hallucination/test.csv', 
    'test_set_May6': '/data2/rsalgani/reddit/test_sets/test_set_May6/specialized_test.csv'
}
audio_dict = {
    'music_caps': ['/data2/rsalgani/music-caps/wav/', 'wav'], 
    'song_describer': ['/data2/rsalgani/song_describer/audio/', 'mp3'],
    'test_set_May6': []
}

def launch():
    args = parse_args()
    res_path = f'/data2/rsalgani/saved_embeddings/{args.eval_data}/{args.gen_model}/'
    id_key = 'track_id' if args.eval_data == 'song_describer' else 'ytid'
    # caption_key = 'caption'
    run_eval(
        og_path = prompt_dict[args.eval_data],
        res_path=res_path, 
        id_key=id_key, 
        table_caption_key=args.table_caption_key, 
        gen_caption_key=args.gen_caption_key,
        dataset=args.eval_data
    )

launch() 
