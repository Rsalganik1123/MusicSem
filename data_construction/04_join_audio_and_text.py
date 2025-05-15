import pandas as pd 
import pickle
import json 
from glob import glob
import torch  
import ipdb 
from tqdm import tqdm 
import shutil

def get_df_idx(fps): 
    idx = [] 
    for fp in fps: 
        data = json.load(open(fp, 'r'))
        idx.append(data['df_idx'])
    return idx 
def df_captions(df, idx):
    mus_cap, context_cap, emotion_cap = [], [], [] 
    for i in idx:  
        df_slice = df.iloc[i, :]
        mus_cap.extend(df_slice.musical_attributes) 
        context_cap.extend(df_slice.contexts_in_which_music_is_listened_to)
        emotion_cap.extend(df_slice.emotions_the_songs_elicit) 

        # cap_set.append(df.musical_captions.tolist()[0], df.contexts_in_which_music_is_listened_to.tolist()[0], df.emotions_the_songs_elicit.tolist()[0])
        
    return mus_cap, context_cap, emotion_cap

def flatten(cap_set): 
    if len(cap_set)>1:
        return [x for xs in cap_set for x in xs.split(' ')]
    else: 
        return cap_set[0].split(' ')

def word_overlap(lmpc_response, cap):
    overlap = len(set(lmpc_response)&set(cap)) /len(set(cap))
    return overlap 

def overlap_runner(lmpc_response, cap_set): 
    # ipdb.set_trace() 
    overlap_metrics = [] 
    for cap in cap_set: 
        if len(cap) == 0: overlap_metrics.append(-1)
        else: 
            overlap_metrics.append(word_overlap(lmpc_response, flatten(cap))) 
    return overlap_metrics  

ipdb.set_trace()
thread = 'LetsTalkMusic'

spotify_jsons = sorted(glob(f'/data2/rsalgani/reddit/spotify_json/{thread}2/*.json'))
response_jsons =  sorted(glob(f'/data2/rsalgani/reddit/json_temps/{thread}/*.json'))

spotify_df = pd.DataFrame({'idx':[int(fp.split('/')[-1].split('.json')[0].split('_')[0]) for fp in spotify_jsons], 'sp_json': spotify_jsons})
response_df = pd.DataFrame({'idx':[int(fp.split('/')[-1].split('.json')[0]) for fp in response_jsons], 'gpt_json': response_jsons})

ipdb.set_trace()
df = pd.merge(spotify_df, response_df, on='idx')


ipdb.set_trace()
 
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

ipdb.set_trace() 
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

long_description = final_df[final_df.d_len > 70.0]

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

ipdb.set_trace() 
print(len(long_description))
# pickle.dump(relevant, open(f'/data2/rsalgani/reddit/collection/relevant_only/{thread}2.pkl', 'wb'))

pickle.dump(super_relevant, open(f'/data2/rsalgani/reddit/collection/super_relevant/{thread}3.pkl', 'wb'))

pickle.dump(long_description, open(f'/data2/rsalgani/reddit/collection/long_description/{thread}3.pkl', 'wb'))


# dfs = []
# for f in files: 
#     dfs.append(pickle.load(open(f, 'rb')))
# all_dfs = pd.concat(dfs)

# sample = final_df.sample(100)
# paths = [] 
# for f in sample.audio_file: 
#     final_name = f.split('/')[0]
#     shutil.copy(f, f'/data2/rsalgani/reddit/old/baby_data2/{final_name}')
#     paths.append(f'/data2/rsalgani/reddit/old/baby_data2/{final_name}') 
# sample['audio_file'] = paths 
# pickle.dump(sample, open('/data2/rsalgani/reddit/old/baby_data2/captions.pkl', 'wb'))

# idx = [] 
# mp3_fps = [] 
# for fp in tqdm(spotify_jsons):
#     descriptor = fp.split('/')[-1].split('.json')[0]
#     df_idx, NER, uri =  descriptor.split('_')
#     data = json.load(open(fp, 'r'))
#     mp3_fps.append(data['audio_fp'])
#     idx.append(int(df_idx)) 
 
# audio_df = pd.DataFrame({'idx': idx, 'audio_fp': mp3_fps})
# audio_df = audio_df.groupby('idx').aggregate(list).reset_index() 
# # ipdb.set_trace()  
# # merged_m = pd.merge(m, audio_df, left_index=True, right_on='idx')
# merged_df = pd.merge(df, audio_df, left_index=True, right_on='idx')

# pickle.dump(merged_df, open('/data2/rsalgani/SemanticMM/data_process_gpt_batches/LetsTalkMusic_responses+music.pkl', 'wb'))

# ipdb.set_trace() 
# lmpc_emb = torch.concatenate(lmpc_emb, dim=0)
# cap_emb = torch.concatenate(cap_emb, dim=0)
# torch.save(lmpc_emb, '/data2/rsalgani/reddit/temp/LetsTalkMusic/sent_emb/lpmc.pt')
# torch.save(cap_emb, '/data2/rsalgani/reddit/temp/LetsTalkMusic/sent_emb/gpt.pt')

# metric_df = pd.DataFrame({'uri':uris, 'pairs':num_examples, 'music_overlap':mus_overlap, 'context_overlap':context_overlap, 'emotion_overlap': emotion_overlap, 'sim':sim1})
# metric_df = metric_df.replace(-1.0, None)
# metric_df = metric_df.dropna(subset =['music_overlap', 'context_overlap', 'emotion_overlap'], how='all')
# metric_df = metric_df.fillna(0.0)
# metric_df['total_overlap'] = metric_df[[m for m in list(metric_df.columns) if m != 'uri']].sum(axis=1)
# print('overlap stats', metric_df.total_overlap.describe())

