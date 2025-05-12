import os 

def musiclm_gen(args, dataset_dict): 
    data_path = dataset_dict[args.dataset]
    command = f"python3 /u/rsalgani/2024-2025/BenchMM/models/t2m/musiclm/open-musiclm/scripts/infer.py \
        --save_dir {args.save_dir} \
        --prompt_path {data_path} \
        --prompt_key {args.prompt_key} \
        --dataset {args.dataset}\
        --semantic_path /data2/rsalgani/clap_checkpoints/semantic.transformer.14000.pt \
        --coarse_path /data2/rsalgani/clap_checkpoints/coarse.transformer.18000.pt \
        --fine_path /data2/rsalgani/clap_checkpoints/fine.transformer.24000.pt \
        --rvq_path /data2/rsalgani/clap_checkpoints/clap.rvq.950_no_fusion.pt \
        --kmeans_path /data2/rsalgani/clap_checkpoints/kmeans_10s_no_fusion.joblib \
        --model_config /data2/rsalgani/clap_checkpoints/musiclm_large_small_context.json \
        --duration 4" 
    os.popen(command).read() 
