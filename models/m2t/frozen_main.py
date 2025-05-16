import argparse
import os 
import warnings  
warnings.filterwarnings("ignore")


prompt_dict = {
    'music_caps': '<path to csv>', 
    'song_describer': '<path to csv>',
    'music_sem': '<path to csv>'
}
audio_dict = {
    'music_caps': ['<path to audio>', 'wav'], 
    'song_describer': ['<path to audio>', 'mp3'],
    'music_sem': ['<path to audio>', 'mp3'],
}


def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_model', choices=['mullama', 'lp_musiccaps', 'futga'])
    parser.add_argument('--dataset', choices = ['music_caps', 'song_describer', 'music_sem'])
    parser.add_argument('--save_dir', default='/data2/rsalgani/saved_embeddings/')
    parser.add_argument('--prompt_key', default='caption')
    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--mode', choices=['classic', 'counterfac'], default='classic')
    return parser.parse_args()

if __name__ == '__main__': 
    args = parse_args() 
    if not os.path.exists(f'{args.save_dir}{args.dataset}/{args.gen_model}/'): 
        os.makedirs(f'{args.save_dir}{args.dataset}/{args.gen_model}/')
    if args.gen_model == 'mullama': #activate mullama env
        from m2t.mullama.mullama import mullama_gen
        mullama_gen(args, audio_dict)
    if args.gen_model == 'futga': #activate futga env
        from m2t.futga.futga import futga_gen
        futga_gen(args, audio_dict)
    if args.gen_model == 'lp_musiccaps': #activate lp_musiccaps env
        from m2t.lpmusiccaps.lpmusiccaps import lpmusiccaps_gen
        lpmc_gen(args, audio_dict)