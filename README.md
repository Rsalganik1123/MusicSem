# MusBench

Our code is structured around 3 key folders: (1) data_construction, (2) models, and (3) evaluations


# Data Construction
This folder consists of all the pieces of our data construction pipeline. 

We provide users with a sample reddit thread containing only 20 entries to showcase the formatting of our extraction pipeline. 

To run the example code, you use the following steps. 

0. Build and activate the conda environment from /u/`environments/data_construct.txt`

1. create a file called ```data_construction/secret_keys.py``` which has the following entries: 
```
spotify_client_secret = ''
spotify_client_id = '' 

openai_key = ''
claude_key = ''
```

Note: you will need to create an account with Spotify developed via this link: https://developer.spotify.com. 

2. Change directories via `cd data_construction/`

3. If you want to use some reddit thread which is not included in the dataset repository, you will start by downloading from https://the-eye.eu/redarcs/ and then running `00_zst_to_pd.py`. However, if you wish to start with one of the thread provided, for example the sample thread in `data/sample_for_testing/GuiltyPleasureMusic` you will start with the code `01_pd_to_prompt.py`. For example, you can run: 
```
python3 01_pd_to_prompt.py --input_dir /MusBench/data/sample_for_testing --thread GuiltyPleasureMusic --output_dir /MusBench/data/sample_for_testing/extract_prompts/
```
This code will extract the raw posts and format them into prompts for the extraction phase. 

4. Next you will run the extraction. This code will feed each raw post to the model and extract the semantic elements using the prompt defined in `extraction_prompt_template.py`. If you wish to include more elements into your extraction, please feel free to add them there. To launch, you can use: 
```
python3 02_prompt_to_extract.py --jsonl_path /MusBench/data/sample_for_testing/extract_prompts/GuiltyPleasureMusic_batches.jsonl --output_dir /MusBench/data/sample_for_testing/extractions/
```
Please note that your code will require you to create your own API with OpenAI. Without this you will not be able to replicate our extraction protocols. 

5. Next, you will gather the audio samples associated with the song,artist pairs extracted from the raw post and verify that they are aligned with the text in the post. To launch, you can use: 
```
python3 03_scrape_audio.py --input_dir /u/rsalgani/2024-2025/MusBench/data/sample_for_testing/extract_outputs/batch_run_20250515_121516/ --output_dir /u/rsalgani/2024-2025/MusBench/data/sample_for_testing/spotify/ --thread GuiltyPleasureMusic

```
Please note that the process of loading mp3 files is the true bottleneck of this entire process. However, it is not necessary to run this all at once. For example, you can asynchronously leave the audios to scrape as you identify the threads that you are interested. That is why we write this code on a thread-by-thread basis. 

6. Once you have extracted the audio, you want to make sure that you combine the raw posts, semantics, and audio. This is because not all the song,artist pairs will be correctly matched to an mp3 file. In our filtration protocol, we remove any raw post and its associated semantics if none of the mp3 files can be recovered (as this is likely an indication of model hallucination). To launch this run: 
```
python3 04_join_audio_and_text.py --audio_dir /u/rsalgani/2024-2025/MusBench/data/sample_for_testing/spotify/spotify_json --thread GuiltyPleasureMusic --input_dir /u/rsalgani/2024-2025/MusBench/data/sample_for_testing/extract_outputs/batch_run_20250515_121516/indiv_jsons --output_dir /u/rsalgani/2024-2025/MusBench/data/sample_for_testing/collection/
```
This will provide you with several collections of (raw_post, semantic collection, mp3) thruples. First, we define atmospheric_only. This section contains the raw posts and associated semantics/mp3s that contain at least the atmospheric elements. Then we define long description for raw posts which contain descriptive elements which are longer than a certain threshold. You can manually edit this threshold but we set it to a string length of 70 characters. In the toy example, we have lowered this threshold so as to maintain some examples for showcasing the entire pipeline. 


7. Next, we use an LLM to generate summaries of the semantic extractions into sentences. You are welcome to skip this step if you are not interested in using LLM-based sentences and would prefer the raw text. To launch, run: 
```
python3 06_create_prompt.py --output_dir /u/rsalgani/2024-2025/MusBench/data/sample_for_testing/summary_prompts --gathered_file /u/rsalgani/2024-2025/MusBench/data/sample_for_testing/collection/gathered.pkl --chunk_size 1

```
Please note that in our example we set a very small chunk size but if you have larger data, we encourage you to increase this. 

8. Next, we run hallucination checks on the semantic summarizations by comparing them with the extracted tags. To run this, use: 
```
python3 07_hallucination_check2.py --gathered_file /u/rsalgani/2024-2025/MusBench/data/sample_for_testing/summary_prompts/20250515_134257/gathered_prompt_for_summary.pkl --output_dir /u/rsalgani/2024-2025/MusBench/data/sample_for_testing/collection
```

9. Finally, we gather all the entries together and format them. To launch, run: 
```
python3 08_final_clean.py --gathered_file /u/rsalgani/2024-2025/MusBench/data/sample_for_testing/collection/clean/FINAL_20250515_135306 --output_dir /u/rsalgani/2024-2025/MusBench/data/sample_for_testing/collection/final
```



# Running Inference

We create a smooth pipeline for users to quickly reproduce the results we showcase in our paper. 
For each model we have a file in the `environments` folder. Before running please build the environment associated with that model name. Unfortunately, we are not resposible for any changes made to the libraries but we have done our best to be as thorough as possible and you are always welcome to reach out to us for help. 

## Text to Music Generation
We present frozen repositories for each of the model implementations and a command line code to run the evaluations. 

To evaluate, activate the associated conda env and then run: 
```
python3 frozen_main.py --gen_model musicgen --dataset music_sem --prompt_key prompt
```

## Music to Text Generaton
We present frozen repositories for each of the model implementations and a command line code to run the evaluations. 

To evaluate, activate the associated conda env and then run: 
```
python3 frozen_main.py --gen_model mullama --dataset music_sem --prompt_key prompt
```

## Cross Modal Retrieval 
We have encapsulated text/audio encoders for LARP, CLAP, ImageBind, and CLaMP3 under the directory retrieval/models/{model_name}/{model_name}_encoder, where {model_name} corresponds to these model identifiers. Our retrieval interface automatically encodes the entire dataset and enables cross-modal querying using either text prompts or audio file paths.

```
python retrieval.py --query "happy pop song with strong drums" --top_k 5 --encoder LARP --dataset_json data/MSD-Eval.json --feature_dir .
```

Automatically checks for pre-extracted features in --feature_dir. If features exist: Loads them directly; If not: Performs fresh feature extraction and saves results.

# Evaluation

## Text to Music Generation

For text-to-music models we provide two different evaluations based on two different toolkits, https://github.com/microsoft/fadtk and https://github.com/haoheliu/audioldm_eval. 

To evaluate with fadtk you can run your evaluation using: 
```
python3 t2m_special_FAD.py --ref_model encodec-emb --ref_dataset fma_pop --gen_model musicgen --eval_data music_caps
```

To evaluate with audiolm_eval you can run your evaluation using: 
```
python3 t2m_cannon.py --ref_dir <path to reference audio directory> --gen_dir <path to generated audio directory> --metric <"clap", "vendi" or "kld"> --csv_dir <path to csv file>
```
## Music to Text Generation
We present frozen repositories for each of the model implementations and a command line code to run the evaluations. For text-to-music models we provide an evaluation based on the toolkit from https://github.com/deezer/playntell. 

To evaluate with playntell you can run your evaluation using: 
```
python3 m2t.py --eval_data music_sem --gen_model mullama --track_id spotify_id --table_caption_key prompt --gen_caption_key caption_as_str --csv_path <path to csv>

```

## Cross Modal Retrieval

We implement the following metrics to evaluate cross-modal retrieval performance:

Recall@1, Recall@5, Recall@10, NDCG@5, NDCG@10, and MRR.

To run the evaluation, use:
```
python3 retrival_eval.py --dataset_path /path/to/dataset.json --encoder LARP --task_name LARP-MSD
```
