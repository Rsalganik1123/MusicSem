import pandas as pd
import os
from utils import parse_args
import datetime 
import pickle 
import ipdb

def main(input_path, output_path): 
    df = pickle.load(open(input_path, 'rb'))
    print(df)
    def get_unique_id(text):
        id = os.path.basename(text).split(".")[0]
        return id

    def get_spotify_url(text):
        # input : /data2/rsalgani/reddit/spotify_mp3/3HfEgAaf0koxBpBB8NvGda.mp3
        # output: https://open.spotify.com/track/3HfEgAaf0koxBpBB8NvGda
        id = os.path.basename(text).split(".")[0]
        url = f"https://open.spotify.com/track/{id}"

        return url

    def get_thread(text):
        # input: /data2/rsalgani/reddit/json_temps/LetsTalkMusic/1000.json
        # output: progrockmusic
        thread = text.split("/")[-2].lower()
        return thread

    keys = [] 
    # ipdb.set_trace()
    df['unique_id'] = df['audio_file'].apply(get_unique_id)
    df['spotify_link'] = df['audio_file'].apply(get_spotify_url)
    df['thread'] = df['gpt_json'].apply(get_thread)
    df = df.rename(columns={'description': 'prompt'})
    df = df[df.hallucination_detected == False]

    # reorder
    cols = [
        "unique_id",
        "thread",
        "spotify_link",
        "song",
        "artist",
        "raw_text",
        "prompt",
        "descriptive",
        "contextual",
        "situational",
        "atmospheric",
        "metadata",
        "pairs"
    ]

    df = df[cols]

    print(df)
    os.makedirs(output_path, exist_ok=True)
    TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_excel(f'{output_path}/final_{TIMESTAMP}.xlsx')

if __name__ == "__main__":
    args = parse_args() 
    main(args.gathered_file, args.output_dir)