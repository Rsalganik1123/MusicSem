import pandas as pd 
import time 
import ipdb 
import argparse
import requests
import urllib3
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException
import os
import pickle
import requests
from tqdm import tqdm
import numpy as np 
import threading
import shutil
import re
import sys
import json 
import shutil
from glob import glob 
import requests
import pickle 
import itertools
import time 
import re
import json


from utils import parse_args
from secret_keys import * 


os.environ['SPOTIPY_CLIENT_ID'] = spotify_client_id
os.environ['SPOTIPY_CLIENT_SECRET'] = spotify_client_secret

def previously_loaded(read_path, json_folder_path, json_responses, good_idx):
    # ipdb.set_trace() 
    written = glob(f'{json_folder_path}*.json') 
    written_idx = sorted([int(fp.split('/')[-1].split('_')[0]) for fp in written])
    all_idx = sorted([int(fp.split('/')[-1].split('.json')[0]) for fp in json_responses]) 
    if len(good_idx) > 0 : 
        all_idx = [idx for idx in all_idx if idx in good_idx] 
    # all_idx = [i for i in all_idx if i > np.max(written_idx)]
    # ipdb.set_trace() 
    if len(written_idx) == 0: 
        last_completed_idx = 0
    else: 
        last_completed_idx = np.max(written_idx)
    to_read = [f'{read_path}{i}.json' for i in all_idx if i not in written_idx and i > last_completed_idx]
    return to_read  

def __find_mp3(name, audio_format):
    # ipdb.set_trace() 
    name = name.strip('?!')
    for i in os.listdir("./"):
        s = name.split(" ")
        if i.endswith(s[-1]+f".{audio_format}") and i.startswith(s[0]):
            return i

def generate_string(string): 
    return string.replace(' ', '%20')

def fact_check(response, query_artist, query_track):
    response_track_name = response['name'].lower() 
    response_artist_name = response['artists'][0]['name'].lower()
    artist_overlap = set(response_artist_name) & set(query_artist)
    track_overlap = set(response_track_name) & set(query_track)
    threshold = (len(artist_overlap) >= .6*len(set(query_artist))) & (len(track_overlap) >= .5*len(set(query_track))) 
    return len(artist_overlap)/len(set(query_artist))*100, len(track_overlap)/len(set(query_track))*100 , threshold
    
def download_youtube_audio(uri: str, spotify_track_name, audio_folder_path, audio_format, sample_rate=48000, verbose=False):
    os.chdir(audio_folder_path)
    if 'spotify:' in uri: 
        uri = uri.split(":")[-1]
    cmd = f"spotdl https://open.spotify.com/track/{uri} --format {audio_format}  --scan-for-songs --overwrite force --ffmpeg-args '-ar {sample_rate}' --cookie /data2/rsalgani/cookie.txt" 
    res = os.popen(cmd).read()
    if "Downloaded" in res:
        # Find the name of the music
        music_name = re.findall(r'"(.*?)"', res)[0]
        file_name = __find_mp3(music_name, audio_format) 
        # os.rename(file_name, os.path.join(audio_folder_path, f"{uri}.{audio_format}")) 
        src_path, dst_path = file_name, os.path.join(audio_folder_path, f"{uri}.{audio_format}")
        dest = shutil.move(src_path, dst_path) 
        if verbose: print(f"moved from {file_name} to {dest}")
        # move this file to save dir
        if not os.path.exists(dst_path): 
            return False 
        return dst_path
    else: 
        print(res)
    return ''

def format_pairs(pair_list): 
    # ipdb.set_trace() 
    if np.array(pair_list).ndim == 1: 
        if type(pair_list[0]) == str: 
            if len(pair_list) == 2:  #Case 1: [song, artist] e.g. ['The Musical Box', 'Genesis'] 
                pair_list = [pair_list]
            if len(pair_list) > 2:   #Case 2: [s1, a1, s2, a2] e.g. ['The Musical Box', 'Genesis']
                pair_list = [pair_list[i:i + 2] for i in range(0, len(pair_list), 2)]
        if type(pair_list[0]) == dict: 
            #Case 4: [{'song': 'Fragile', 'artist': 'unknown'}, {'song': 'Close to the Edge', 'artist': 'unknown'}]
            new_list = []
            for p in pair_list: 
                new_list.append([p['song'], p['artist']])
            return new_list 
        return pair_list 
        
    if np.array(pair_list).ndim == 2: #Case 3: [['s1', 'a1']]
        return pair_list  

def search_uri_from_NER(sp, json_fp, audio_folder_path, json_folder_path, pairs_list): 
    good, bad = 0, 0
    fps = []  
    idx = json_fp.split('/')[-1].split('.json')[0]
    pairs_list = format_pairs(pairs_list)
    # ipdb.set_trace() 
    for pair in pairs_list: 
        try: 
            song, artist = pair[0], pair[1]
        except Exception as e:
            print(e, pair)

        # query_format = f'{song.lower()}%20{artist.lower()}'.replace(' ', '%20')
        query_format=f'{song.lower()} {artist.lower()}'
        try: 
            time.sleep(1)
            response = sp.search(q=query_format, type='track', market='US', limit=5)
            # artist_info = requests.get('https://api.spotify.com/v1/search?q={}&type={}'.format(artist_name, 'artist'), header = {'access_token': access_token})
            top_response = response['tracks']['items'][0]
            spotify_uri = top_response['uri'].split(':')[-1]
            spotify_track_name = top_response['name']
            artist_overlap, song_overlap, flag = fact_check(top_response, artist, song)
            if flag:
                if os.path.exists(f'{audio_folder_path}{spotify_uri}.mp3'): 
                    write_path = f'{audio_folder_path}{spotify_uri}.mp3'
                else: 
                    write_path = download_youtube_audio(spotify_uri, spotify_track_name, audio_folder_path, 'mp3')
                if len(write_path)> 1: 
                    data = {
                        'response_fp': json_fp, 
                        'artist': artist, 
                        'song': song, 
                        'spotify_uri': spotify_uri, 
                        'audio_fp': write_path, 
                        'hallucination_score': (artist_overlap, song_overlap),
                        'below_threshold': flag
                    }
                    json.dump(data, open(f'{json_folder_path}{idx}_{good}_{spotify_uri}.json', 'w'))
                    fps.append(f'{json_folder_path}{idx}_{good}_{spotify_uri}.json')
                    good +=1 
        except SpotifyException as e:
            if e.http_status == 429:
                print("'retry-after' value:", e.headers['retry-after'])
                retry_value = e.headers['retry-after']
                if int(e.headers['retry-after']) > 200:
                    print("STOP FOR TODAY, retry value too high {}".format(retry_value))
                    exit() 
                else:
                    time.sleep(retry_value)
                    continue
        except Exception as e:
            print(e)
            continue 
        
    return good, bad,  fps 

def sanitize_json_like_string(raw_str):
    """
    Converts a malformed JSON-like string (e.g., with Python tuples, escape chars)
    into valid JSON that can be loaded with json.loads.
    """
    # Remove leading/trailing backticks and 'json' marker if present
    cleaned = raw_str.strip().strip("`")
    
    # Replace Python tuples (e.g., ("a", "b")) with JSON lists (["a", "b"])
    cleaned = re.sub(r'\(\s*"(.*?)"\s*,\s*"(.*?)"\s*\)', r'["\1", "\2"]', cleaned)
    cleaned = re.sub(r"\(\s*'(.*?)'\s*,\s*'(.*?)'\s*\)", r'["\1", "\2"]', cleaned)

    # Replace single quotes with double quotes (except inside words like "Plain White T's")
    # Use a smarter quote replace: only replace keys and values
    def replace_single_quotes(match):
        return f'"{match.group(1)}"'

    cleaned = re.sub(r"'([^']*)'", replace_single_quotes, cleaned)

    # Remove escape characters like \n
    cleaned = cleaned.replace('\\n', '').replace('\\', '')

    # Final trim and parse
    cleaned = cleaned.strip()
    return json.loads(cleaned)

def split_jsonl_into_json(input_dir, thread): 
    ipdb.set_trace()  
    data = json.load(open(f'{input_dir}processed_extractions_output/extracted_data.json', 'r'))
    df = pd.DataFrame(data[f'{thread}_batches'])
    keys = ['pairs', 'Descriptive', 'Contextual', 'Situational', 'Atmospheric', 'Metadata', 'idx']
    df = df.rename(columns={"input": "raw_text", 'raw_text':"random", 'custom_id':'idx'}) 
    df.idx = df.idx.apply(lambda x: int(x.split('-')[-1])) 
    output_dir = f'{input_dir}indiv_jsons/'
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
    rel_idx = [] 
    for i in tqdm(range(len(df))): 
        rel_count = 0 
        row = df.iloc[i, : ]
        
        if row['pairs'] != []: rel_count += 1 
        else: 
            continue 
        if row['Descriptive'] != []: rel_count += 1 
        if row['Contextual'] != []: rel_count += 1 
        if row['Situational'] != []: rel_count += 1 
        if row['Atmospheric'] != []: rel_count += 1 
        if row['Metadata'] != []: rel_count += 1 
        if rel_count >=3 :
            j = row[keys].to_json()
            j_data = {
                    'extraction_model': 'gpt-4', 
                    'raw_post': row['raw_text'], 
                    'extraction': j }
            json.dump(j_data, open(f'{output_dir}/{i}.json', 'w'))
            rel_idx.append(i)
    print(len(rel_idx))
    return output_dir

def main(input_dir:str, thread:str, output_dir, good_idx=[]): 
    session = requests.Session()
    retry = urllib3.Retry(
        respect_retry_after_header=False
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)

    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(),
                        requests_session=session)
    audio_folder_path = f'{output_dir}spotify_mp3/'
    json_folder_path = f'{output_dir}spotify_json/{thread}/'
    response_folder_path = f'{input_dir}'
    os.makedirs(json_folder_path, exist_ok=True)
    os.makedirs(audio_folder_path, exist_ok=True)
    all_json_responses = glob(f'{response_folder_path}*.json') 
    json_responses = previously_loaded(response_folder_path,json_folder_path, all_json_responses, good_idx)
    good, bad, all_writes = 0, 0, 0 
    idx, mp3_files = [], []  

    with tqdm(total=len(json_responses), desc='') as pbar:
        for fp in json_responses: 
            temp = json.load(open(fp, 'r'))
            data = json.loads(temp['extraction'])
            try: 
                if len(data['pairs']) == 0: 
                    bad += 1
                    pbar.update()
                    pbar.set_postfix(**{'keep':good, 'dump': bad, 'ratio of keeps': good/len(json_responses)})
                    continue
                else: 
                    # ipdb.set_trace()
                    print(data['pairs']) 
                    wrote, dumped, files = search_uri_from_NER(sp, fp, audio_folder_path, json_folder_path, data['pairs'])
                    all_writes += wrote
                    df_idx = fp.split('/')[-1].split('.json')[0]
                    idx.extend([df_idx]*len(files))
                    mp3_files.extend(files)
                    if wrote >=1: 
                        good += 1 
                    else: bad += 1 
            except Exception as e:
                print(e) 
                bad += 1
                pbar.update()
                pbar.set_postfix(**{'keep':good, 'dump': bad, 'ratio of keeps': good/len(json_responses)})
                continue
            pbar.update()
            pbar.set_postfix(**{'+':good, '-': bad, 'keep ratio': good/len(json_responses), 'mp3':all_writes})
        df = pd.DataFrame({'df_idx': idx, 'mp3_files':mp3_files})
        pickle.dump(df, open(f'/data2/rsalgani/reddit/org/{thread}_mp3_file_locations.pkl', 'wb'))

if __name__ == '__main__': 
    args = parse_args()
    json_path = split_jsonl_into_json(args.input_dir, args.thread)
    print(json_path)
    main(input_dir=json_path,
        thread=args.thread, 
        output_dir=args.output_dir)