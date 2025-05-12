import requests
import wget
import shutil 
import pandas as pd 
import time 
import ipdb 
from tqdm import tqdm 
from glob import glob 
import json 
import pickle 
import os 


###MUREKA CREDS#### 
account = "69626583056385"
token = "8KJ70IUwnJ7zWKkkOFRG1NDyK6cI8B9k"
api_token = "user:1707-sDItUMvHnJzgrxpaipHEk"

def mureka_gen(args, dataset_dict): 
    data_path = dataset_dict[args.dataset]

    if 'pkl' in data_path: 
        df = pickle.load(open(data_path, 'rb'))
    elif 'csv' in data_path: 
        df = pd.read_csv(data_path)
    else: 
        raise NotImplemented 
    
    previously_read = [int(f.split('/')[-1].split('.mp3')[0]) for f in glob(f"{args.save_dir}{args.dataset}/mureka/*.mp3")]
    if len(previously_read) < 1: 
        start = 0 
    else: 
        start = max(previously_read)

    for i in tqdm(range(start, len(df))): 
        row = df.iloc[i, :]
        prompt = row[args.prompt_key]
        if len(prompt)> 300: #mureka can only sustain prompts with less than 300 tokens. 
            prompt=prompt[:250]
        apiUrl = f"https://api.useapi.net/v1/mureka/music/create" 
        headers = {
            "Content-Type": "application/json", 
            "Authorization" : f"Bearer {api_token}"
        }
        body = {
            "account": f"{account}",
            "prompt": f"{prompt}"
        }
        try: 
            response = requests.post(apiUrl, headers=headers, json=body)
            print(response)
            print(response.json())
            # ipdb.set_trace() 
            if not os.path.exists(f'/data2/rsalgani/reddit/json_mureka/{args.dataset}/'): 
                os.makedirs(f'/data2/rsalgani/reddit/json_mureka/{args.dataset}/')
            json.dump(response.json(), open(f'/data2/rsalgani/reddit/json_mureka/{args.dataset}/{i}.json', 'w'))
            src = wget.download(response.json()['songs'][0]['mp3_url'])
            dst = f'{args.save_dir}{args.dataset}/mureka/{i}.mp3'
            shutil.move(src, dst)
        except: 
            continue 
        time.sleep(5)
        if i % 10: 
            time.sleep(20)



        # if i % 33:
        #     time.sleep(50)


# def mureka_gen(args, dataset_dict): 
#     account = "69626583056385"
#     token = "8KJ70IUwnJ7zWKkkOFRG1NDyK6cI8B9k"
#     api_token = "user:1707-sDItUMvHnJzgrxpaipHEk"
#     def check_token(): 
#         apiUrl = f"https://api.useapi.net/v1/mureka/accounts/{account}" 
#         headers = {
#             "Content-Type": "application/json", 
#             "Authorization" : f"Bearer {api_token}"
#         }
#         body = {
#             "account": f"{account}",
#             "token": f"{token}"
#         }
#         response = requests.post(apiUrl, headers=headers, json=body)
#         print(response, response.json())

#     def prompt_api(): 
#         data = pd.read_csv('/data2/rsalgani/song_describer/song_describer.csv')
#         already_read = glob('/data2/rsalgani/reddit/json_mureka/song_describer/*')
#         start_idx = max([int(idx.split('/')[-1].split('.json')[0]) for idx in already_read])
#         for i in tqdm(range(start_idx+1, len(data))): 
#             row = data.iloc[i, :]
#             prompt = row.caption
#             if len(prompt)> 300: 
#                 prompt=prompt[:200]
#             apiUrl = f"https://api.useapi.net/v1/mureka/music/create" 
#             headers = {
#                 "Content-Type": "application/json", 
#                 "Authorization" : f"Bearer {api_token}"
#             }
#             body = {
#                 "account": f"{account}",
#                 "prompt": f"{prompt}"
#             }
#             response = requests.post(apiUrl, headers=headers, json=body)
#             print(response)
#             print(response.json())
#             json.dump(response.json(), open(f'/data2/rsalgani/reddit/json_mureka/song_describer/{i}.json', 'w'))
#             src = wget.download(response.json()['songs'][0]['mp3_url'])
#             dst = f'/data2/rsalgani/saved_embeddings/song_describer/mureka/{i}.mp3'
#             shutil.move(src, dst)
#             time.sleep(5)
#             if i % 10: 
#                 time.sleep(20)
#             if i % 33:
#                 time.sleep(50)