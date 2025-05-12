from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy
import os 
import pandas as pd  
import ipdb 
from tqdm import tqdm 
import pickle
from glob import glob  

os.environ['HF_HOME']="/data2/rsalgani/hub/"


# ipdb.set_trace() 
# df = pd.read_csv('/data2/rsalgani/music-caps/musiccaps-public.csv')
# df = pd.read_csv('/data2/rsalgani/song_describer/song_describer.csv')


def musicgen_gen(args, dataset_dict): 

    device = args.device

    processor = AutoProcessor.from_pretrained("facebook/musicgen-large")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-large").to(device)

    data_path = dataset_dict[args.dataset]

    if 'pkl' in data_path: 
        df = pickle.load(open(data_path, 'rb'))
    elif 'csv' in data_path: 
        df = pd.read_csv(data_path)
    else: 
        raise NotImplemented 
    # ipdb.set_trace() 
    previously_read = [int(f.split('/')[-1].split('.wav')[0]) for f in glob(f"{args.save_dir}{args.dataset}/musicgen/*.wav")]
    if len(previously_read) < 1: 
        start = 0 
    else: 
        start = max(previously_read)
    start = 400 
    print(f'previously loaded:{start}')
    for i in tqdm(range(start, len(df))): 
        row = df.iloc[i, :]
        caption = row[args.prompt_key]
        try:
            inputs = processor(
                    text=[caption],
                    padding=True,
                    return_tensors="pt",
                ).to(device)
        except: 
            print(caption)
            continue 
        audio_values = model.generate(**inputs, max_new_tokens=1000).to(device)
        sampling_rate = model.config.audio_encoder.sampling_rate
        scipy.io.wavfile.write(f"{args.save_dir}{args.dataset}/musicgen/{i}.wav", rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())

    # df = pickle.load(open('/data2/rsalgani/counterfac2/org2.pkl', 'rb'))
    # keys = ['original','Descriptive', 'Situational', 'Atmospheric', 'Metadata', 'Contexutal']

    # for i in tqdm(range(len(df))): 
    #     row = df.iloc[i, :]
    #     idx = row.idx 
    #     try: 
    #         for k in tqdm(keys): 
    #             caption = row[k]
    #             if k == "Contexutal": 
    #                 ipdb.set_trace() 
    #             if caption != 'None':
    #                 inputs = processor(
    #                     text=[caption],
    #                     padding=True,
    #                     return_tensors="pt",
    #                 ).to(device)

    #                 audio_values = model.generate(**inputs, max_new_tokens=1000).to(device)
    #                 sampling_rate = model.config.audio_encoder.sampling_rate
    #                 scipy.io.wavfile.write(f"/data2/rsalgani/saved_embeddings/counterfac2/musicgen/{idx}_{k}.wav", rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())
    #     except Exception as e: 
    #         print(e)