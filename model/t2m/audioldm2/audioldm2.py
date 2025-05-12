from diffusers import AudioLDM2Pipeline
import torch
import scipy
import pandas as pd 
from tqdm import tqdm 
import pickle 
from glob import glob 

def audioldm2_gen(args, dataset_dict): 
    repo_id = "cvssp/audioldm2"
    pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
    pipe = pipe.to(args.device)

    data_path = dataset_dict[args.dataset]

    if 'pkl' in data_path: 
        df = pickle.load(open('/data2/rsalgani/counterfac2/org2.pkl', 'rb'))
    elif 'csv' in data_path: 
        df = pd.read_csv(data_path)
    else: 
        raise NotImplemented 
    previously_read = [int(f.split('/')[-1].split('.wav')[0]) for f in glob(f"{args.save_dir}{args.dataset}/audioldm2/*.wav")]
    if len(previously_read) < 1: 
        start = 0 
    else: 
        start = max(previously_read)
    print(f'previously loaded:{start}')
    # start = 300 
    for i in tqdm(range(start, len(df))): 
        row = df.iloc[i, :]
        prompt = row[args.prompt_key]
        try:
            if len(prompt) < 5: 
                raise Exception
            audio = pipe(prompt, num_inference_steps=200, audio_length_in_s=10.0).audios[0]
            scipy.io.wavfile.write(f"{args.save_dir}{args.dataset}/audioldm2/{i}.wav", rate=16000, data=audio)
        except: 
            print(prompt)
            continue 

def audioldm2_counterfac_test(args, dataset_dict): 
    repo_id = "cvssp/audioldm2"
    pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
    pipe = pipe.to(args.device)

    data_path = dataset_dict[args.dataset]

    if 'pkl' in data_path: 
        df = pickle.load(open('/data2/rsalgani/counterfac2/org2.pkl', 'rb'))
    elif 'csv' in data_path: 
        df = pd.read_csv(data_path)
    else: 
        raise NotImplemented 

    for i in tqdm(range(len(df))): 
        row = df.iloc[i, :]
        prompt = row[args.prompt_key]
        audio = pipe(prompt, num_inference_steps=200, audio_length_in_s=10.0).audios[0]
        scipy.io.wavfile.write(f"{args.save_dir}{args.dataset}/audioldm2/{i}.wav", rate=16000, data=audio)
