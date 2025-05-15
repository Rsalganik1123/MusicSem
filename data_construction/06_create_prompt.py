import anthropic
import yaml
from openai import OpenAI
from summary_prompt_template import DESCRIPTION_PROMPT_BECCA
from copy import deepcopy
import os
import json
import pickle
import pandas as pd
from glob import glob
from tqdm import tqdm
import time
import math
import datetime
from utils import parse_args
from secret_keys import * 

# MODEL = "claude-3-7-sonnet-20250219"
# MODEL = "gpt-4o"
# CHUNK_SIZE = 10000  # Number of rows per batch chunk

# Generate timestamp for this run
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# output_dr = f"/u/rsalgani/2024-2025/MusBench/data/sample_for_testing/generate_description/{TIMESTAMP}"
# os.makedirs(RUN_DIR, exist_ok=True)

# with open("creds.yaml", "r") as f:
#     creds = yaml.safe_load(f)
#     anthropic_api_key = creds["Anthropic_API_KEY"]
#     openai_api_key = creds["OPENAI_API_KEY"]

def generate_description(prompt_template, max_tokens=300, temperature=1, params=None, model='gpt-4o'):
    prompt = deepcopy(prompt_template)

    if params is None:
        params = {}
    prompt = prompt.format(**params)

    
    client = OpenAI(api_key=openai_key)
    response = client.chat.completions.create(
        model='gpt-4o',
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )
    description = response.choices[0].message.content


    return description

def split_dataframe_into_chunks(df, chunk_size):
    """Split the dataframe into chunks of specified size"""
    return [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

def prepare_batch_requests(df_chunk):
    """Prepare batch requests for a chunk of dataframe rows"""
    batch_requests = []
    idx_mapping = {}  # Map custom_ids to row indices
    
    for i, (idx, row) in enumerate(df_chunk.iterrows()):
        input_tags = {
            "descriptive": row.get("descriptive", []),
            "situational": row.get("situational", []),
            "atmospheric": row.get("atmospheric", []),
            "metadata": row.get("metadata", []),
            "lyrical": row.get("lyrical", [])
        }
        
        prompt = deepcopy(DESCRIPTION_PROMPT_BECCA)
        prompt = prompt.format(input_tags=input_tags)
        
        custom_id = f"request-{i}"
        idx_mapping[custom_id] = idx
        
        batch_requests.append(json.dumps({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": 'gpt-4o',
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
                "temperature": 1
            }
        }))
    
    return batch_requests, idx_mapping

def process_batch_chunk(chunk_id, df_chunk, file_prefix, output_dir):
    """Process a single chunk of the dataframe using the batch API"""
    
    client = OpenAI(api_key=openai_key)
    
    # Create status file path in timestamped directory
    status_dir = os.path.join(output_dir, "batch_status")
    os.makedirs(status_dir, exist_ok=True)
    status_file = os.path.join(status_dir, f"{file_prefix}_chunk_{chunk_id}_status.json")
    
    # Check if this chunk was already processed successfully
    if os.path.exists(status_file):
        with open(status_file, "r") as f:
            status_data = json.load(f)
            if status_data.get("status") == "completed":
                print(f"Chunk {chunk_id} already processed successfully. Skipping.")
                return True, "Already processed", None
    
    # Prepare batch requests
    batch_requests, idx_mapping = prepare_batch_requests(df_chunk)
    
    # Write batch requests to file in timestamped directory
    chunk_dir = os.path.join(output_dir, "batch_chunk")
    os.makedirs(chunk_dir, exist_ok=True)
    batch_file_path = os.path.join(chunk_dir, f"{file_prefix}_batch_requests_chunk_{chunk_id}.jsonl")
    with open(batch_file_path, "w") as f:
        f.write("\n".join(batch_requests))
    
    # Upload the file
    batch_file = client.files.create(
        file=open(batch_file_path, "rb"),
        purpose="batch"
    )
    
    # Create the batch
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    
    print(f"Batch for chunk {chunk_id} created with ID: {batch.id}")
    
    # Save initial status
    with open(status_file, "w") as f:
        json.dump({
            "chunk_id": chunk_id,
            "batch_id": batch.id,
            "status": "in_progress",
            "row_count": len(df_chunk),
            "started_at": time.time()
        }, f)
    
    # Poll for completion
    max_retries = 100  # chec
    for retry in range(max_retries):
        try:
            batch_status = client.batches.retrieve(batch.id)
            print(f"Chunk {chunk_id} status: {batch_status.status} ({retry+1}/{max_retries})")
            
            if batch_status.status == "completed":
                output_file_id = batch_status.output_file_id
                
                # Download results
                output_content = client.files.content(output_file_id)
                results = [json.loads(line) for line in output_content.text.strip().split("\n")]
                
                # Create a new dataframe for processed results
                result_df = df_chunk.copy()
                
                # Process results
                success_count = 0
                failure_count = 0
                
                for result in results:
                    custom_id = result["custom_id"]
                    idx = idx_mapping[custom_id]
                    
                    if result["error"] is not None:
                        print(f"Error processing row {idx}: {result['error']}")
                        failure_count += 1
                        continue
                    
                    try:
                        description = result["response"]["body"]["choices"][0]["message"]["content"]
                        result_df.at[idx, "description"] = description
                        success_count += 1
                    except Exception as e:
                        print(f"Error saving result for row {idx}: {str(e)}")
                        failure_count += 1
                
                # Update status file
                with open(status_file, "w") as f:
                    json.dump({
                        "chunk_id": chunk_id,
                        "batch_id": batch.id,
                        "status": "completed",
                        "success_count": success_count,
                        "failure_count": failure_count,
                        "completed_at": time.time()
                    }, f)
                
                return True, f"Processed {success_count} rows successfully, {failure_count} failures", result_df
                
            elif batch_status.status in ["failed", "expired", "cancelled"]:
                # Update status file with failure
                with open(status_file, "w") as f:
                    json.dump({
                        "chunk_id": chunk_id,
                        "batch_id": batch.id,
                        "status": batch_status.status,
                        "error": f"Batch {batch_status.status}",
                        "completed_at": time.time()
                    }, f)
                return False, f"Batch {batch_status.status}", None
            
            # Wait before checking again (10 minutes)
            time.sleep(15)
            
        except Exception as e:
            print(f"Error checking batch status: {str(e)}")
            # Continue trying despite errors
    
    # If we get here, it's a timeout
    with open(status_file, "w") as f:
        json.dump({
            "chunk_id": chunk_id,
            "batch_id": batch.id,
            "status": "timeout",
            "error": "Timeout waiting for batch completion",
            "completed_at": time.time()
        }, f)
    return False, "Timeout waiting for batch completion", None

def process_individual(df_chunk):
    """Process each row in the dataframe individually"""
    result_df = df_chunk.copy()
    
    for idx, row in tqdm(df_chunk.iterrows()):
        try:
            input_tags = {
                "descriptive": row.get("descriptive", []),
                "situational": row.get("situational", []),
                "atmospheric": row.get("atmospheric", []),
                "metadata": row.get("metadata", []),
                "lyrical": row.get("lyrical", [])
            }
            
            description = generate_description(DESCRIPTION_PROMPT_BECCA, 
                                              params={"input_tags": input_tags})
            result_df.at[idx, "description"] = description
            
        except Exception as e:
            print(f"Error processing row {idx} individually: {str(e)}")
    
    return result_df

def check_progress(file_prefixes):
    """Check the overall progress of all chunks across all files"""
    status_dir = os.path.join(args.output_dir, "batch_status")
    if not os.path.exists(status_dir):
        return {"total": 0, "completed": 0, "failed": 0, "in_progress": 0}
    
    # Collect status files for all prefixes
    status_files = []
    for prefix in file_prefixes:
        status_files.extend(glob(os.path.join(status_dir, f"{prefix}_chunk_*_status.json")))
    
    total = len(status_files)
    completed = 0
    failed = 0
    in_progress = 0
    
    for status_file in status_files:
        with open(status_file, "r") as f:
            status_data = json.load(f)
            
            if status_data.get("status") == "completed":
                completed += 1
            elif status_data.get("status") in ["failed", "expired", "cancelled", "timeout"]:
                failed += 1
            else:
                in_progress += 1
    
    return {
        "total": total,
        "completed": completed,
        "failed": failed,
        "in_progress": in_progress
    }

if __name__ == "__main__":
    args = parse_args() 
    TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = f'{args.output_dir}/{TIMESTAMP}'

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Keep track of all file prefixes for progress reporting
    file_prefixes = []
    
    
    with open(args.gathered_file, "rb") as f:
        data = pickle.load(f)
    
    
    os.makedirs(args.output_dir, exist_ok=True)

    # Split the dataframe into chunks
    df_chunks = split_dataframe_into_chunks(data, args.chunk_size)
    total_chunks = len(df_chunks)
    print(f"Split {len(data)} rows into {total_chunks} chunks")
    
    # Create a new dataframe to store all results
    final_df = pd.DataFrame()

    file_prefix = 'gathered'

    # Process each chunk
    for chunk_id, chunk in enumerate(df_chunks):
        print(f"Processing chunk {chunk_id+1}/{total_chunks} with {len(chunk)} rows")
        

        # Try batch processing first
        success, message, result_df = process_batch_chunk(chunk_id, chunk, file_prefix, args.output_dir)
        
        if not success or result_df is None:
            print(f"Chunk {chunk_id} batch processing failed: {message}")
            print(f"Falling back to individual processing for chunk {chunk_id}")
            result_df = process_individual(chunk)
        
        # Append results to final dataframe
        final_df = pd.concat([final_df, result_df])
        
        # Save intermediate results in timestamped directory
        # temp_dir = os.path.join(args.output_dir, "temp")
        # os.makedirs(temp_dir, exist_ok=True)
        # temp_output_path = os.path.join(temp_dir, f"temp_{file_prefix}_chunk_{chunk_id}.pkl")
        # with open(temp_output_path, "wb") as f:
        #     pickle.dump(result_df, f)
        
        # Show overall progress for batch processing
        
        progress = check_progress(file_prefixes)
        print(f"Overall progress: {progress['completed']}/{progress['total']} chunks completed, {progress['failed']} failed, {progress['in_progress']} in progress")
    
    with open(f'{args.output_dir}/gathered_with_summary.pkl', "wb") as f:
        pickle.dump(final_df, f)
    
    print(f"Processing completed for {args.gathered_file}. Results saved to {args.output_dir}")
    
