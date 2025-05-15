import pandas as pd 
import pickle 
from utils import parse_args 
from tqdm import tqdm 
import json 
import os 

from extraction_prompt_template import get_prompt 


def build_jsonl(input_dir, thread, output_dir, model): 
    df = pickle.load(open(f'{input_dir}/{thread}/{thread}_joined.pkl', 'rb'))
    if not os.path.exists(f'{output_dir}'): 
        os.makedirs(f'{output_dir}')
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


if __name__ == '__main__':
    """
    Note: This code will load from joint dataframe and add column with prompts 
            Prompt based on strategy specified in parser
    Returns:
        pandas dataframe: joint dataframe and add column with prompts 
    """ 
    args = parse_args()
    build_jsonl(input_dir=args.input_dir, thread = args.thread, output_dir=args.output_dir, model='gpt-4o')