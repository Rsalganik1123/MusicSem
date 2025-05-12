import soundfile as sf
import os 
# from mustango_repo.mustango import Mustango
import pandas as pd 
from tqdm import tqdm 
from glob import glob 
import pickle
import sys 
sys.path.append('/u/rsalgani/2024-2025/MusBench/model/t2m/mustango/mustango')
from mustango import Mustango 

def mustango_gen(args, dataset_dict): 
    device = args.device 
    model = Mustango("declare-lab/mustango") 
    data_path = dataset_dict[args.dataset]

    if 'pkl' in data_path: 
        df = pickle.load(open(data_path, 'rb'))
    elif 'csv' in data_path: 
        df = pd.read_csv(data_path)
    else: 
        raise NotImplemented 
    
    for i in tqdm(range(len((df)))): 
        row = df.iloc[i, :]
        cap = row[args.prompt_key]
        try: 
            music = model.generate(cap)
            sf.write(f"{args.save_dir}{args.dataset}/mustango/{i}.wav", music, samplerate=16000)
        except Exception as e: 
            print(e)
            print(i, cap)
            continue  

        
    