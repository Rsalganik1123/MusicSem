

import pandas as pd 
import pickle
import json 
from glob import glob
import torch  
import ipdb 
from tqdm import tqdm 
import shutil



# thread = 'LetsTalkMusic'

# spotify_jsons = sorted(glob(f'/data2/rsalgani/reddit/spotify_json/{thread}2/*.json'))
# response_jsons =  sorted(glob(f'/data2/rsalgani/reddit/json_temps/{thread}/*.json'))

# spotify_df = pd.DataFrame({'idx':[int(fp.split('/')[-1].split('.json')[0].split('_')[0]) for fp in spotify_jsons], 'sp_json': spotify_jsons})
# response_df = pd.DataFrame({'idx':[int(fp.split('/')[-1].split('.json')[0]) for fp in response_jsons], 'gpt_json': response_jsons})

# ipdb.set_trace()
# df = pd.merge(spotify_df, response_df, on='idx')


# ipdb.set_trace()
 
# data = []

# for idx in tqdm(range(len(df))): 
#     row = df.iloc[idx, :]
#     gpt = json.load(open(row.gpt_json, 'r'))
#     raw_text = gpt['raw_post']
#     extraction = json.loads(gpt['extraction'])
#     try: 
#         all_pairs = extraction['pairs']
#         descriptive = extraction['Descriptive']
#         contextual = extraction['Contextual']
#         situational = extraction['Situational']
#         atmospheric = extraction['Atmospheric']
#         # lyrical = extraction['Lyrical']
#         metadata = extraction['Metadata']
#     except: 
#         continue 
#     sp = json.load(open(row.sp_json, 'r'))
#     artist, song, hallucination_score, audio_fp = sp['artist'], sp['song'], sp['hallucination_score'], sp['audio_fp']
#     data.append({
#         'idx':row.idx, 
#         'sp_json': row.sp_json, 
#         'gpt_json': row.gpt_json,
#         'audio_file': audio_fp, 
#         'raw_text': raw_text, 
#         'song': song, 
#         'artist': artist, 
#         'hallucination_score_song': hallucination_score[0], 
#         'hallucination_score_artist': hallucination_score[1], 
#         'descriptive': descriptive, 
#         'contextual': contextual, 
#         'situational': situational, 
#         'atmospheric': atmospheric, 
#         # 'lyrical': lyrical, 
#         'metadata': metadata,
#         'pairs': all_pairs
#     })

 

ipdb.set_trace() 
files = ['/data2/rsalgani/reddit/collection/super_relevant/electronicmusic.pkl', 
         '/data2/rsalgani/reddit/collection/super_relevant/LetsTalkMusic3.pkl', 
         '/data2/rsalgani/reddit/collection/super_relevant/musicsuggestions2.pkl', 
         '/data2/rsalgani/reddit/collection/super_relevant/popheads.pkl', 
         '/data2/rsalgani/reddit/collection/super_relevant/progrockmusic.pkl', 
         '/data2/rsalgani/reddit/collection/long_description/electronicmusic.pkl', 
         '/data2/rsalgani/reddit/collection/long_description/LetsTalkMusic3.pkl', 
         '/data2/rsalgani/reddit/collection/long_description/musicsuggestions2.pkl', 
         '/data2/rsalgani/reddit/collection/long_description/popheads.pkl', 
         '/data2/rsalgani/reddit/collection/long_description/progrockmusic.pkl']

dfs = [] 
for f in files: 
    df = pickle.load(open(f, 'rb'))
    dfs.append(df)

df = pd.concat(dfs)
df = df[~df['song'].str.contains('Plastic Love')]
df = df[~df['artist'].str.contains('Billyrrom')]
print(len(df))

# print(len(df))
# min_length = 100
# max_length = 1000
# df = df[df['raw_text'].str.len() >= min_length]
# df = df[df['raw_text'].str.len() <= max_length]


super_relevant = df[
    (df.descriptive.str.len() != 0) &
    (df.atmospheric.str.len() != 0) &
    (df.situational.str.len() != 0)
]


print('total length', len(df))
print('super relevant', len(super_relevant))
print('descriptive', len(df[df.descriptive.str.len() != 0].descriptive)/ len(df)) 
print('atmospheric', len(df[df.atmospheric.str.len() != 0].atmospheric)/len(df)) 
print('situational', len(df[df.situational.str.len() != 0].situational)/len(df)) 
print('contextual', len(df[df.contextual.str.len() != 0].contextual) /len(df)) 
print('metadata', len(df[df.metadata.str.len() != 0].metadata)/len(df) ) 


pickle.dump(df, open('/data2/rsalgani/reddit/collection/final/total_May9.pkl', 'wb')) 
pickle.dump(super_relevant, open('/data2/rsalgani/reddit/collection/final/sr_May9.pkl', 'wb')) 
df.to_excel('/data2/rsalgani/reddit/collection/final/total_May9.xlsx')
print(len(df))