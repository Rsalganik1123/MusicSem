import os 

def futga_gen(args, dataset_dict): 
    data_path = dataset_dict[args.dataset]
    command = f"python3 /u/rsalgani/2024-2025/MusBench/model/m2t/futga/FUTGA/evaluation.py \
        --dataset {args.dataset}\
        --audio_path {data_path[0]}\
        --audio_format {data_path[1]}\
        --save_dir {args.save_dir} "
    os.popen(command).read() 
