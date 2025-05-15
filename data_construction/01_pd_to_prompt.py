import pandas as pd 
import pickle 
from utils import parse_args 

from prompt_template import get_prompt 

def build(fp:str, thread:str, prompt_strategy:str): 
    df = pickle.load(open(f'{fp}joined_data/{thread}_joined.pkl', 'rb'))
    prompts = [] 
    with tqdm(total = len(df), desc=f'running prompting') as pbar: 
        for i in range(len(df)):
            row = df.iloc[i, :]
            body = row.body 
            prompt_base, prompt_end = get_prompt(prompt_strategy)
            prompts.append(prompt_base + body + prompt_end) 
    final_df = df[f'P_{prompt_strategy}']
    pickle.dump(final_df, open(f'{fp}joined_w_prompts/{thread}_{prompt_strategy}.pkl', 'rb'))

if __name__ == '__main__':
    """
    Note: This code will load from joint dataframe and add column with prompts 
            Prompt based on strategy specified in parser
    Returns:
        pandas dataframe: joint dataframe and add column with prompts 
    """ 
    args = parse_args()
    build(args.fp, args.thread, args.prompt_strategy)