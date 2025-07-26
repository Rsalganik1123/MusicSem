import csv
import json

def musiccaps_csv_to_json(csv_path):
    eval_data = []

    with open(csv_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        for row in csv_reader:
            entry = {
                "_id": f"musicCaps-{row['ytid']}",
                "caption": row['caption'],
                "wave_path": f"/data/tteng/MuLM/musicCaps_wave_data/{row['ytid']}.wav"
            }

            eval_data.append(entry)

    with open('musicCaps-Eval.json', 'w', encoding='utf-8') as eval_file:
        json.dump(eval_data, eval_file, indent=4, ensure_ascii=False)


def songdescriber_csv_to_json(csv_path):
    train_data = []
    eval_data = []

    with open(csv_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        for row in csv_reader:
            entry = {
                "_id": f"songDescriber-{row['track_id']}",
                "caption": row['caption'],
                "wave_path": f"/data/tteng/MuLM/songDescriber_wave_data/{row['path'][:-3] + '2min.mp3'}"
            }
            
            eval_data.append(entry)
                
    with open('songDescriber-Eval.json', 'w', encoding='utf-8') as eval_file:
        json.dump(eval_data, eval_file, indent=4, ensure_ascii=False)


def smd_csv_to_json(csv_path):
    json_data = []

    with open(csv_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        for row in csv_reader:
            audio_file_path = row['audio_file_1m']
            file_id = audio_file_path.split('/')[-1].split('.')[0]
            
            entry = {
                "_id": f"SMD-{file_id}",
                "caption": row['prompt'],
                "wave_path": f"/data/tteng/MuLM/SMD/1min_audio/{file_id}.mp3"
            }
            json_data.append(entry)
    
    with open('MSD-Eval.json', 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    musiccaps_csv_to_json('musiccaps-public.csv')
    songdescriber_csv_to_json('song_describer.csv')
    smd_csv_to_json("test_May6_final.csv")