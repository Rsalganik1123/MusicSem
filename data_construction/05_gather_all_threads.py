

import pandas as pd 
import pickle
import json 
from glob import glob
import torch  
import ipdb 
from tqdm import tqdm 
import shutil
from utils import parse_args



def main(input_dir, output_dir): 
    # ipdb.set_trace() 
    files = glob(f'{input_dir}/long_description/*.pkl')
    sr_files = glob(f'{input_dir}/super_relevant/*.pkl')
    at_files = glob(f'{input_dir}/atmospheric_only/*.pkl')
    files.extend(sr_files)
    files.extend(at_files)
    
    dfs = [] 
    for f in files: 
        df = pickle.load(open(f, 'rb'))
        dfs.append(df)

    df = pd.concat(dfs)
    df = df[~df['song'].str.contains('Plastic Love')]
    df = df[~df['artist'].str.contains('Billyrrom')]
    print(len(df))


    super_relevant = df[
        (df.descriptive.str.len() != 0) &
        (df.atmospheric.str.len() != 0) &
        (df.situational.str.len() != 0)
    ]


    print('total length', len(df))
    print('super relevant', len(super_relevant))
    print('descriptive', len(df[df.descriptive.str.len() != 0].descriptive)/ len(df)) 
    print('atmospheric', len(df[df.atmospheric.str.len() != 0].atmospheric)/len(df)) 
    print('situational', len(df[df.situational.str.len() != 0].situational)/len(df)) 
    print('contextual', len(df[df.contextual.str.len() != 0].contextual) /len(df)) 
    print('metadata', len(df[df.metadata.str.len() != 0].metadata)/len(df) ) 

    pickle.dump(df, open(f'{output_dir}/gathered.pkl', 'wb')) 
    # df.to_excel(f'{output_dir}.xlsx')
    # print(len(df))

if __name__ == '__main__': 
    args = parse_args()
    main(args.input_dir, args.output_dir)
