import anthropic
import yaml
from openai import OpenAI
from hallucination_prompt_template import HALLUCINATION_PROMPT_BECCA
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
import re
from utils import parse_args
from secret_keys import * 


MODEL = "claude-3-7-sonnet-20250219"
# MODEL = "gpt-4o"
# CHUNK_SIZE = 10000  # Number of rows per batch chunk

# Create timestamp for this run
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# with open("creds.yaml", "r") as f:
#     creds = yaml.safe_load(f)
#     anthropic_api_key = creds["Anthropic_API_KEY"]
#     openai_api_key = creds["OPENAI_API_KEY"]

def generate_description(prompt_template, max_tokens=4096, temperature=1, params=None, model=MODEL):
    prompt = deepcopy(prompt_template)

    if params is None:
        params = {}
    prompt = prompt.format(**params)

    if model.startswith("gpt"):
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        description = response.choices[0].message.content

    elif model.startswith("claude"):
        client = anthropic.Anthropic(api_key=claude_key)
        message = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        )
        description = message.content[0].text
    else:
        raise ValueError(f"Invalid model: {model}")

    return description

def split_dataframe_into_chunks(df, chunk_size):
    """Split the dataframe into chunks of specified size"""
    return [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

def prepare_batch_requests(df_chunk):
    # """Prepare batch requests for a chunk of dataframe rows"""
    # batch_requests = []
    # idx_mapping = {}  # Map custom_ids to row indices
    
    # for i, (idx, row) in enumerate(df_chunk.iterrows()):
    #     input_tags = {
    #         "descriptive": row.get("descriptive", []),
    #         "situational": row.get("situational", []),
    #         "atmospheric": row.get("atmospheric", []),
    #         "lyrical": row.get("lyrical", [])
    #     }
        
    #     prompt = deepcopy(DESCRIPTION_PROMPT_BECCA)
    #     prompt = prompt.format(input_tags=input_tags)
        
    #     custom_id = f"request-{i}"
    #     idx_mapping[custom_id] = idx
        
    #     batch_requests.append(json.dumps({
    #         "custom_id": custom_id,
    #         "method": "POST",
    #         "url": "/v1/chat/completions",
    #         "body": {
    #             "model": MODEL,
    #             "messages": [{"role": "user", "content": prompt}],
    #             "max_tokens": 300,
    #             "temperature": 1
    #         }
    #     }))
    
    # return batch_requests, idx_mapping
    pass

def prepare_batch_requests_claude(df_chunk, template):
    """Prepare batch requests for Claude's Batch API"""
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
        
        prompt = deepcopy(template)
        prompt = prompt.format(features=input_tags, 
                               summary=row["description"],
                               raw_text = row["raw_text"])
        
        custom_id = f"request-{i}"
        idx_mapping[custom_id] = idx
        
        batch_requests.append({
            "custom_id": custom_id,
            "params": {
                "model": MODEL,
                "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}]
            }
        })
    
    return batch_requests, idx_mapping

def process_batch_chunk(chunk_id, df_chunk):
    """Process a single chunk of the dataframe using the batch API"""
    if not MODEL.startswith("gpt"):
        return False, "Batch API only works with OpenAI models"
        
    client = OpenAI(api_key=openai_api_key)
    
    # Create status file path with timestamp
    status_dir = os.path.join(f"batch_status_hallucination_{TIMESTAMP}")
    os.makedirs(status_dir, exist_ok=True)
    status_file = os.path.join(status_dir, f"pkl_chunk_{chunk_id}_status.json")
    
    # Check if this chunk was already processed successfully
    if os.path.exists(status_file):
        with open(status_file, "r") as f:
            status_data = json.load(f)
            if status_data.get("status") == "completed":
                print(f"Chunk {chunk_id} already processed successfully. Skipping.")
                return True, "Already processed", None
    
    # Prepare batch requests
    batch_requests, idx_mapping = prepare_batch_requests(df_chunk)
    
    # Write batch requests to file with timestamp
    chunk_dir = os.path.join(f"batch_chunk_{TIMESTAMP}")
    os.makedirs(chunk_dir, exist_ok=True)
    batch_file_path = f"{chunk_dir}/pkl_batch_requests_chunk_{chunk_id}.jsonl"
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
    max_retries = 100  # check for up to ~100 minutes
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
            
            # Wait before checking again (1 minute)
            time.sleep(10)
            
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

def process_batch_chunk_claude(chunk_id, df_chunk, base_filename, template):
    """Process a single chunk of the dataframe using Claude's Batch API"""
    if not MODEL.startswith("claude"):
        return False, "Batch API only works with Claude models", None
        
    client = anthropic.Anthropic(api_key=claude_key)
    
    # # Create status file path with timestamp
    status_dir = os.path.join(f"{args.output_dir}/cache_check_hallucination/{TIMESTAMP}/batch_status_hallucination")
    os.makedirs(status_dir, exist_ok=True)
    status_file = os.path.join(status_dir, f"claude_chunk_{base_filename}_{chunk_id}_status.json")
    
    # Check if this chunk was already processed successfully
    if os.path.exists(status_file):
        with open(status_file, "r") as f:
            status_data = json.load(f)
            if status_data.get("status") == "completed":
                print(f"Chunk {chunk_id} already processed successfully. Skipping.")
                return True, "Already processed", None
    
    # Prepare batch requests
    batch_requests, idx_mapping = prepare_batch_requests_claude(df_chunk, template)
    
    # Create the batch
    message_batch = client.messages.batches.create(
        requests=batch_requests
    )
    
    print(f"Batch for chunk {chunk_id} created with ID: {message_batch.id}")
    
    # Save initial status
    with open(status_file, "w") as f:
        json.dump({
            "chunk_id": chunk_id,
            "batch_id": message_batch.id,
            "status": "in_progress",
            "row_count": len(df_chunk),
            "started_at": time.time()
        }, f)

    # Save batch requests to file
    batch_dir = os.path.join(f"cache_check_hallucination/{TIMESTAMP}/batch_requests_hallucination")
    os.makedirs(batch_dir, exist_ok=True)
    batch_file_path = os.path.join(batch_dir, f"claude_batch_requests_chunk__{base_filename}_{chunk_id}.json")
    with open(batch_file_path, "w") as f:
        json.dump(batch_requests, f)
    
    # Poll for completion
    max_retries = 100  # check for up to ~100 minutes
    for retry in range(max_retries):
        try:
            batch_status = client.messages.batches.retrieve(message_batch.id)
            print(f"Chunk {chunk_id} status: {batch_status.processing_status} ({retry+1}/{max_retries})")
            
            if batch_status.processing_status == "ended":
                # Create a new dataframe for processed results
                result_df = df_chunk.copy()
                
                # Process results
                success_count = 0
                failure_count = 0
                
                # Stream results
                for result in client.messages.batches.results(message_batch.id):
                    custom_id = result.custom_id
                    idx = idx_mapping[custom_id]
                    
                    if result.result.type == "succeeded":
                        hallucination = result.result.message.content[0].text
                        result_df.at[idx, "check_hallucination"] = hallucination
                        success_count += 1
                    else:
                        print(f"Error processing row {idx}: {result.result.type}")
                        failure_count += 1
                
                # Update status file
                with open(status_file, "w") as f:
                    json.dump({
                        "chunk_id": chunk_id,
                        "batch_id": message_batch.id,
                        "status": "completed",
                        "success_count": success_count,
                        "failure_count": failure_count,
                        "completed_at": time.time()
                    }, f)
                
                return True, f"Processed {success_count} rows successfully, {failure_count} failures", result_df
                
            # Wait before checking again (1 minute)
            time.sleep(100)
            
        except Exception as e:
            print(f"Error checking batch status: {str(e)}")
            # Continue trying despite errors
    
    # If we get here, it's a timeout
    with open(status_file, "w") as f:
        json.dump({
            "chunk_id": chunk_id,
            "batch_id": message_batch.id,
            "status": "timeout",
            "error": "Timeout waiting for batch completion",
            "completed_at": time.time()
        }, f)
    return False, "Timeout waiting for batch completion", None

def process_individual(df_chunk):
    # """Process each row in the dataframe individually"""
    # result_df = df_chunk.copy()
    
    # for idx, row in tqdm(df_chunk.iterrows()):
    #     try:
    #         input_tags = {
    #             "descriptive": row.get("descriptive", []),
    #             "situational": row.get("situational", []),
    #             "atmospheric": row.get("atmospheric", []),
    #             "lyrical": row.get("lyrical", [])
    #         }
            
    #         description = generate_description(DESCRIPTION_PROMPT_BECCA, 
    #                                           params={"input_tags": input_tags})
    #         result_df.at[idx, "description"] = description
            
    #     except Exception as e:
    #         print(f"Error processing row {idx} individually: {str(e)}")
    
    # return result_df
    pass

def check_progress():
    """Check the overall progress of all chunks"""
    status_dir = os.path.join(f"batch_status_{TIMESTAMP}")
    if not os.path.exists(status_dir):
        return {"total": 0, "completed": 0, "failed": 0, "in_progress": 0}
    
    status_files = glob(os.path.join(status_dir, "pkl_chunk_*_status.json"))
    
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

def extract_hallucination_result(df):
    """
    Parse the check_hallucination column to extract the hallucination_detected JSON value.
    Adds a new column 'hallucination_detected' with True/False values.
    
    Args:
        df: DataFrame containing 'check_hallucination' column
    
    Returns:
        DataFrame with added 'hallucination_detected' column
    """
    if 'check_hallucination' not in df.columns:
        return df
    
    def parse_response(response):
        if not isinstance(response, str):
            return None
            
        # Find the JSON block containing hallucination_detected
        json_match = re.search(r'```\s*\{[\s\n]*"hallucination_detected":\s*(true|false)[\s\n]*\}\s*```', response, re.IGNORECASE)
        
        if json_match:
            # Extract and parse the JSON
            try:
                json_str = json_match.group(0).strip('`').strip()
                result = json.loads(json_str)
                return result.get('hallucination_detected')
            except json.JSONDecodeError:
                pass
                
        # Fallback method: look for true/false after "hallucination_detected":
        fallback_match = re.search(r'"hallucination_detected":\s*(true|false)', response, re.IGNORECASE)
        if fallback_match:
            value = fallback_match.group(1).lower()
            return value == 'true'
            
        return None
    
    # Apply parsing function
    df['hallucination_detected'] = df['check_hallucination'].apply(parse_response)
    return df

if __name__ == "__main__":
    args = parse_args()
    # Path to the folder containing input pickle files
    # data_folder = "/data2/rsalgani/hallucination/generate_description/long_description"

    # Get all pickle files in the folder
    # pkl_files = glob(os.path.join(data_folder, "*.pkl"))
# 
    # pkl_files = ["/data2/rsalgani/hallucination/generate_description/super_relevant/musicsuggestions2_filter.pkl"]
    # pkl_files = ["/data2/rsalgani/hallucination/generate_description/long_description/musicsuggestions2_filter.pkl"]
    # pkl_files = ["/data2/rsalgani/hallucination/generate_description/super_relevant/LetsTalkMusic2.pkl"]
    # pkl_files = ["/data2/rsalgani/hallucination/generate_description/long_description/LetsTalkMusic2.pkl"]
    # pkl_files = ["/data2/rsalgani/hallucination/generate_description/20250513_052806/json_descriptions_batch_pkl/output/filtered_outer_merge.pkl"]


    # print(f"Found {len(pkl_files)} pickle files to process")
    
    # Create output directory with timestamp
    # output_dir = os.path.join(f"../reddit/json_check_hallucination/{TIMESTAMP}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each pickle file
    
    # print(f"\nProcessing file {file_index+1}/{len(pkl_files)}: {os.path.basename(data_path)}")
    
    # Load the pickle file
    with open(args.gathered_file, "rb") as f:
        data = pickle.load(f)
    
    # Optional: limit data for testing
    # data = data[:10]
    
    # Output path for the processed pickle file
    # base_filename = os.path.basename(data_path)

    final_output_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_output_dir, exist_ok=True)
    TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(final_output_dir, f"FINAL_{TIMESTAMP}.pkl")

    
    
    # Split the dataframe into chunks
    df_chunks = split_dataframe_into_chunks(data, args.chunk_size)
    total_chunks = len(df_chunks)
    print(f"Split {len(data)} rows into {total_chunks} chunks")
    
    # Create a new dataframe to store results for this file
    final_df = pd.DataFrame()
    
    # Process each chunk
    for chunk_id, chunk in enumerate(df_chunks):
        print(f"Processing chunk {chunk_id+1}/{total_chunks} with {len(chunk)} rows")
        
        
        if MODEL.startswith("claude"):
            # Use Claude batch processing
            success, message, result_df = process_batch_chunk_claude(chunk_id, chunk, 'gathered_with_summary', HALLUCINATION_PROMPT_BECCA)
        else:
            success = False
            message = "Unsupported model for batch processing"
            result_df = None
        
        if not success or result_df is None:
            print(f"Chunk {chunk_id} batch processing failed: {message}")
            print(f"Falling back to individual processing for chunk {chunk_id}")
            result_df = process_individual(chunk)
        
        # Append results to final dataframe
        final_df = pd.concat([final_df, result_df])
        
        
        # Show overall progress
        progress = check_progress()
        print(f"Overall progress: {progress['completed']}/{progress['total']} chunks completed, {progress['failed']} failed, {progress['in_progress']} in progress")
    
    # Apply hallucination detection extraction
    final_df = extract_hallucination_result(final_df)
    
    # Save final results for this file
    with open(output_path, "wb") as f:
        pickle.dump(final_df, f)
    
    # print(f"Processing completed for {base_filename}. Results saved to {output_path}")

    # print(f"All files processed. Results saved to {output_dir}")
