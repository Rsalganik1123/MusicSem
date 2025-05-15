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

