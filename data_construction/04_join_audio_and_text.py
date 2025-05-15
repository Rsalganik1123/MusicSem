import pandas as pd 
import pickle
import json 
from glob import glob
import torch  
import ipdb 
from tqdm import tqdm 
import shutil
import os 
from utils import parse_args

def main(thread, spotify_dir, json_dir, output_dir): 
    spotify_jsons = sorted(glob(f'{spotify_dir}/{thread}/*.json'))
    response_jsons =  sorted(glob(f'{json_dir}/*.json'))

    spotify_df = pd.DataFrame({'idx':[int(fp.split('/')[-1].split('.json')[0].split('_')[0]) for fp in spotify_jsons], 'sp_json': spotify_jsons})
    response_df = pd.DataFrame({'idx':[int(fp.split('/')[-1].split('.json')[0]) for fp in response_jsons], 'gpt_json': response_jsons})

    df = pd.merge(spotify_df, response_df, on='idx')

    data = []

    for idx in tqdm(range(len(df))): 
        row = df.iloc[idx, :]
        gpt = json.load(open(row.gpt_json, 'r'))
        raw_text = gpt['raw_post']
        extraction = json.loads(gpt['extraction'])
        try: 
            all_pairs = extraction['pairs']
            descriptive = extraction['Descriptive']
            contextual = extraction['Contextual']
            situational = extraction['Situational']
            atmospheric = extraction['Atmospheric']
            # lyrical = extraction['Lyrical']
            metadata = extraction['Metadata']
        except: 
            continue 
        sp = json.load(open(row.sp_json, 'r'))
        artist, song, hallucination_score, audio_fp = sp['artist'], sp['song'], sp['hallucination_score'], sp['audio_fp']
        data.append({
            'idx':row.idx, 
            'sp_json': row.sp_json, 
            'gpt_json': row.gpt_json,
            'audio_file': audio_fp, 
            'raw_text': raw_text, 
            'song': song, 
            'artist': artist, 
            'hallucination_score_song': hallucination_score[0], 
            'hallucination_score_artist': hallucination_score[1], 
            'descriptive': descriptive, 
            'contextual': contextual, 
            'situational': situational, 
            'atmospheric': atmospheric, 
            # 'lyrical': lyrical, 
            'metadata': metadata,
            'pairs': all_pairs
        })

    
    final_df = pd.DataFrame(data)
    def get_len(x):
        try:  
            if not x: 
                return 0
            elif type(x) == dict: 
                return 0  
            else: 
                return len(" ".join(x))
        except: return 0 

    final_df['d_len'] = final_df.descriptive.apply(lambda x: get_len(x))
    print(final_df[final_df.descriptive.str.len() != 0].descriptive) 
    print(final_df[final_df.atmospheric.str.len() != 0].atmospheric)
    print(final_df[final_df.situational.str.len() != 0].situational) 

    long_description = final_df[final_df.d_len > 1.0]

    relevant = final_df[
        (final_df.descriptive.str.len() != 0) |
        (final_df.atmospheric.str.len() != 0) |
        (final_df.situational.str.len() != 0)
    ]
    super_relevant = final_df[
        (final_df.descriptive.str.len() != 0) &
        (final_df.atmospheric.str.len() != 0) &
        (final_df.situational.str.len() != 0)
    ]

    atmospheric_only = final_df[final_df.atmospheric.str.len() != 0]
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)

    print(len(super_relevant), len(long_description), len(atmospheric_only)) 

    os.makedirs(f'{output_dir}/super_relevant/', exist_ok=True)
    os.makedirs(f'{output_dir}/atmospheric_only/', exist_ok=True)
    os.makedirs(f'{output_dir}/long_description/', exist_ok=True)
    pickle.dump(super_relevant, open(f'{output_dir}super_relevant/{thread}.pkl', 'wb'))
    pickle.dump(long_description, open(f'{output_dir}long_description/{thread}.pkl', 'wb'))
    pickle.dump(atmospheric_only, open(f'{output_dir}atmospheric_only/{thread}.pkl', 'wb'))


if __name__ == '__main__': 
    args = parse_args()
    main(args.thread, args.audio_dir, args.input_dir, args.output_dir)