

# "python3 generate_mullama.py"

import os 

def mullama_gen(args, dataset_dict): 
    data_path = dataset_dict[args.dataset]

    # current_directory = os.getcwd()
    # os.chdir(f'/u/rsalgani/2024-2025/BenchMM/mod /els/t2m/musiclm/open-musiclm/')
    command = f"python3 /u/rsalgani/2024-2025/BenchMM/models/m2t/mullama/MU-LLaMA/ModelEvaluations/generate_mullama.py \
        --dataset {args.dataset}\
        --audio_path {data_path[0]}\
        --audio_format {data_path[1]}\
        --save_dir {args.save_dir} \
        --model /data2/rsalgani/mullama_checkpoint/MU-LLaMA/checkpoint.pth\
        --knn /data2/rsalgani/mullama_checkpoint/MU-LLaMA/ \
        --llama_type 7B \
        --llama_dir /data2/rsalgani/mullama_checkpoint/MU-LLaMA/LLaMA/ \
        --mert_path /data2/rsalgani/hub/models--m-a-p--MERT-v1-330M "
    os.popen(command).read() 