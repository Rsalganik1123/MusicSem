import argparse
import os 
import warnings  
warnings.filterwarnings("ignore")


prompt_dict = {
    'music_caps': '/data2/rsalgani/music-caps/musiccaps-public.csv', 
    'song_describer': '/data2/rsalgani/song_describer/song_describer.csv',
    'test_set_May3': '/data2/rsalgani/hallucination/test.csv', 
    'test_set_May6': '/data2/rsalgani/reddit/test_sets/test_set_May6/package/test_May6_final.csv'
}
audio_dict = {
    'music_caps': ['/data2/rsalgani/music-caps/wav/', 'wav'], 
    'song_describer': ['/data2/rsalgani/song_describer/audio/', 'mp3'],
    'test_set_May6': []
}


def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_model', choices=['musicgen', 'stable_audio', 'musiclm', 'mustango', 'audioldm2', 'mureka'])
    parser.add_argument('--dataset', choices = ['music_caps', 'song_describer', 'test_set_May6'])
    parser.add_argument('--save_dir', default='/data2/rsalgani/saved_embeddings/')
    parser.add_argument('--prompt_key', default='caption')
    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--mode', choices=['classic', 'counterfac'], default='classic')
    return parser.parse_args()

if __name__ == '__main__': 
    args = parse_args() 
    if not os.path.exists(f'{args.save_dir}{args.dataset}/{args.gen_model}/'): 
        os.makedirs(f'{args.save_dir}{args.dataset}/{args.gen_model}/')
    if args.gen_model == 'mureka': #activate mustango env 
        from t2m.mureka.mureka import mureka_gen 
        mureka_gen(args, prompt_dict)
    if args.gen_model == 'mustango': #activate mustango env 
        from t2m.mustango.mustango import mustango_gen 
        mustango_gen(args, prompt_dict)
    if args.gen_model == 'audioldm2': #activate stable_audio2 env 
        from t2m.audioldm2.audioldm2 import audioldm2_gen 
        audioldm2_gen(args, prompt_dict)
    if args.gen_model == 'stable_audio': #activate stable_audio env
        from t2m.stable_audio.stable_audio import stable_audio_gen
        stable_audio_gen(args, prompt_dict)
    if args.gen_model == 'musiclm': #activate musiclm env
        from t2m.musiclm.musiclm import musiclm_gen
        musiclm_gen(args, prompt_dict)
    if args.gen_model == 'musicgen': #activate musicgen env
        from t2m.musicgen.musicgen import musicgen_gen
        musicgen_gen(args, prompt_dict)
    if args.gen_model == 'mullama': #activate mullama env
        from m2t.mullama.mullama import mullama_gen
        mullama_gen(args, audio_dict)
    if args.gen_model == 'futga': #activate futga env
        from m2t.futga.futga import futga_gen
        futga_gen(args, audio_dict)