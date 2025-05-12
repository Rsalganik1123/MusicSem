import pandas as pd 
from glob import glob 
import os 
import re 
import shutil
import argparse  
import ipdb 

def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['eval', 'gather'])
    parser.add_argument('--ref_model', choices= ['encodec-emb', 'MERT-v1-95M', 'vggish'])  
    parser.add_argument('--ref_data', choices=['fma_pop', 'music_caps'], default='fma_pop')
    parser.add_argument('--eval_data', choices=['song_describer', 'music_caps', 'test_set_May6'])
    parser.add_argument('--gen_model', choices=['musicgen', 'stable_audio', 'musiclm', 'mustango', 'audioldm2'])
    parser.add_argument('--mc_path', default='/data2/rsalgani/music-caps/wav/')
    parser.add_argument('--audio_save_dir', default='/data2/rsalgani/saved_embeddings/')
    parser.add_argument('--save_dir', default='/data2/rsalgani/eval_outputs/fad/')
    return parser.parse_args()

def launch(args):
    # ipdb.set_trace() 
    for m in ['musicgen', 'stable_audio', 'musiclm', 'mustango', 'audioldm2', 'mureka']: 
        ref_model = args.ref_model
        mc_path = args.mc_path
        ref_dataset = 'fma_pop' if args.ref_data == 'fma_pop' else mc_path
        exp_dataset = args.eval_data 
        gen_model = m 
        embed_path = f'{args.audio_save_dir}{exp_dataset}/{gen_model}'
        
        cmd =  f"fadtk {ref_model} {ref_dataset} {embed_path} {gen_model}/ --ind"
        exp_folder = f'{args.save_dir}{args.ref_data}/{ref_model}/{exp_dataset}/'
        if not os.path.exists(exp_folder): 
            os.makedirs(exp_folder, exist_ok=True)
        os.chdir(exp_folder)
        res = os.popen(cmd).read()
        if ".csv" in res:
            ipdb.set_trace() 
            src_path = re.findall(r'\b[\w\-]+\.csv\b', res)[0]
            dst_path = f'{exp_folder}fad-{gen_model}.csv'
            dest = shutil.move(src_path, dst_path)
        
def gather(args):  
    ipdb.set_trace() 
    reference_data = args.ref_data
    reference_model = args.ref_model
    dataset = args.eval_data
    path = f'{args.save_dir}{reference_data}/{reference_model}/{dataset}/*'
    exp = glob(path)
    all_models = [] 
    for e in exp: 
        model_name = e.split('/')[-1]
        stats = pd.read_csv(e, names=['fp', 'fad'])
        stats['model'] = model_name
        all_models.append(stats)
    all_stats = pd.concat(all_models).groupby(['model'])['fad'].aggregate(['mean', 'std'])
    print(all_stats)
    
if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'eval': 
        launch(args)
    if args.mode == 'gather': 
        gather(args)
