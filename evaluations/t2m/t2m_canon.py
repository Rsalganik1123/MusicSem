import argparse  

from t2m_toolkit.eval_clap import run_clap_eval 
from t2m_toolkit.eval_ldm import run_ldm_eval 
from t2m_toolkit.eval_vendi import run_vendi_eval 

def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_dir', default='/data2/rsalgani/saved_embeddings/music_caps/audioldm2')
    parser.add_argument('--ref_dir', default='/data2/rsalgani/music-caps/wav')
    parser.add_argument('--csv_dir', default="/data2/rsalgani/music-caps/musiccaps-public.csv")
    parser.add_argument('--metric', default="clap", choices=['clap', 'kld', 'vendi'])
    return parser.parse_args()
    
if __name__ == "__main__": 
    args = parse_args() 
    if args.metric == 'clap': 
        run_clap_eval(args)
    if args.metric == 'kld': 
        run_ldm_eval(args)
    if args.metric == 'vendi': 
        run_vendi_eval(args)
