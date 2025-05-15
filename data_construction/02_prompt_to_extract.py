import pandas as pd 
import pickle 
import json 
from tqdm import tqdm 
import ipdb 
import time 
from openai import OpenAI
import anthropic  
import json 
import ast 
import simplejson
from tqdm import tqdm
import jsonlines
import string 
from glob import glob 
import pickle 
import os 

from utils import parse_args 
from prompt_template import get_prompt
from secret import * 


gpt_client = OpenAI(api_key = openai_key)
claude_client = anthropic.Anthropic(api_key=claude_key)

def check_useful(response): 
    try: 
        response = json.loads(response)
    except Exception as e:
        return False
    non_empty = 0  
    for k in response: 
        if response[k] != []: 
            non_empty += 1 
    try: 
        if len(response['pairs']) == 0: 
            return False 
    except: 
        pass 
    if non_empty == 0: 
        return False 
    return True

def gpt(messages, version): 
    completion = gpt_client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages = messages, 
        response_format={"type": "json_object"})
    response = completion.choices[0].message.content
    return response 

def extract(model, post): 
    if model == 'gpt-4o':
        messages = [ {"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": f"{post}"}]
        response = gpt(messages, 'gpt-4')
        return response 

def df_to_json_for_loop_run(input_dir, thread, output_dir, model:str): 
    df = pickle.load(open(f'{input_dir}{thread}_joined.pkl', 'rb'))
    raw_post, extraction, files, idx = [], [] , [], []  
    read_files = glob(f'{output_dir}json_temps/{thread}/*.json')
    read_idx = [int(f.split('/')[-1].split('.json')[0]) for f in read_files] 
    if len(read_idx) == 0 : start = 0 
    else: start = max(read_idx)
    print(f"already loaded: {start}, starting from where we left off")
    good, bad = 0, 0 
    with tqdm(total = len(df)-start, desc=f'running prompting') as pbar:  
        for i in range(start, len(df)): 
            row = df.iloc[i, :]
            body = f'{row.title} {row.selftext} {row.body}'        
            prompt = get_prompt(body)
            response = extract(model, prompt)
            if not check_useful(response): 
                bad += 1 
                pbar.update()
                pbar.set_postfix(**{'keep':good, 'dump': bad, 'ratio of keeps': good/len(df)})
                continue 
            else: 
                raw_post.append(body)
                extraction.append(response)
                data = {
                    'extraction_model': model, 
                    'raw_post': body, 
                    'extraction': response }
                os.makedirs(f'{output_dir}json_temps/{thread}/', exist_ok=True)
                fp = f'{output_dir}json_temps/{thread}/{i}.json'
                json.dump(data, open(fp, 'w'))
                good += 1 
                idx.append(i)
                files.append(fp)
            time.sleep(1)
            pbar.update()
            pbar.set_postfix(**{'keep':good, 'dump': bad, 'ratio of keeps': good/len(df)})
    output_df = pd.DataFrame({
        'model':[model]*len(files),
        'idx':idx,
        'json_fp': files, 
        'raw_text':raw_post, 
        'extraction': extraction}) 
    pickle.dump(output_df, open(f'{output_dir}/{thread}.pkl', 'wb'))
    return 0 

def df_to_jsonl(input_dir, thread, output_dir, model): 
    df = pickle.load(open(f'{input_dir}{thread}_joined.pkl', 'rb'))
    
    with tqdm(total = len(df), desc=f'writing jsonl') as pbar:  
        with open(output_dir + f'{thread}_batches.jsonl', 'w', encoding='utf-8') as f: 
            for i in range(len(df)): 
                row = df.iloc[i, :]
                body = f'{row.title} {row.selftext} {row.body}'        
                prompt = get_prompt(body)
                data = { 
                        "custom_id": f"request-{i}", 
                        "method": "POST", 
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": f"{model}",
                            "messages": [
                                {"role": "user", "content": f"{prompt}"}],"max_tokens": 1000
                            }
                        }
                f.write(json.dumps(data) + '\n')
                pbar.update() 

    return 0 

df_to_jsonl(input_dir='/data2/rsalgani/reddit/joined_data/', 
            thread='LetsTalkMusic', 
            output_dir = '/data2/rsalgani/reddit/jsonl/',
            model = 'gpt-4o')

# df_to_json_for_loop_run(input_dir = '/data2/rsalgani/reddit/joined_data/', 
#             thread='electronicmusic', 
#             output_dir='/data2/rsalgani/reddit/',  
#             model='gpt-4o')
