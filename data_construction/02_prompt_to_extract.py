import os
import json
import time
import datetime
from openai import OpenAI
import yaml
import argparse
from tqdm import tqdm
import glob
from secret_keys import *
from utils import parse_args

# Default model - we're only using OpenAI since batch API is OpenAI-specific
# MODEL = "gpt-4o"
# POLL_INTERVAL = 120  # seconds between status checks
# CHUNK_SIZE = 10000  # Requests per chunk

def create_time_based_dirs(output_dir):
    """Create time-based directories for this batch run"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(output_dir, f"batch_run_{timestamp}")
    
    # Create directories
    dirs = {
        "status": os.path.join(base_dir, "status"),
        "output": os.path.join(base_dir, "output"),
        "logs": os.path.join(base_dir, "logs"),
        "chunks": os.path.join(base_dir, "chunks")
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return timestamp, dirs

def split_jsonl_into_chunks(jsonl_path, output_dir, chunk_size):
    """Split a JSONL file into multiple smaller chunks"""
    chunk_paths = []
    base_filename = os.path.basename(jsonl_path)
    job_name = os.path.splitext(base_filename)[0]
    
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    num_chunks = (total_lines + chunk_size - 1) // chunk_size  # Ceiling division
    
    print(f"Splitting {total_lines} requests into {num_chunks} chunks...")
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_lines)
        chunk_lines = lines[start_idx:end_idx]
        
        chunk_path = os.path.join(output_dir, f"{job_name}_chunk_{i+1}.jsonl")
        with open(chunk_path, 'w') as f:
            f.writelines(chunk_lines)
        
        chunk_paths.append(chunk_path)
    
    return chunk_paths

def process_jsonl_batch(jsonl_path, batch_dirs, poll_interval, max_retries=100):
    """Process a JSONL file using OpenAI's batch API"""
    client = OpenAI(api_key=openai_key)
    
    # Get the base filename
    base_filename = os.path.basename(jsonl_path)
    job_name = os.path.splitext(base_filename)[0]
    
    # Status file path
    status_file = os.path.join(batch_dirs["status"], f"{job_name}_status.json")
    
    # Check if already processed
    if os.path.exists(status_file):
        with open(status_file, "r") as f:
            status_data = json.load(f)
            if status_data.get("status") == "completed":
                print(f"Chunk {job_name} already processed successfully. Skipping.")
                return True, status_data.get("output_file")
    
    # Upload the JSONL file
    print(f"Uploading file {jsonl_path} for batch processing...")
    batch_file = client.files.create(
        file=open(jsonl_path, "rb"),
        purpose="batch"
    )
    
    # Create the batch
    print(f"Creating batch job for {job_name}...")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    
    batch_id = batch.id
    print(f"Batch created with ID: {batch_id}")
    
    # Save initial status
    with open(status_file, "w") as f:
        json.dump({
            "job_name": job_name,
            "batch_id": batch_id,
            "status": "in_progress",
            "started_at": time.time(),
            "input_file": jsonl_path
        }, f)
    
    # Poll for completion
    for retry in range(max_retries):
        try:
            batch_status = client.batches.retrieve(batch_id)
            status = batch_status.status
            print(f"Batch {job_name} status: {status} ({retry+1}/{max_retries})")
            
            # Update status file with current status
            with open(status_file, "r") as f:
                status_data = json.load(f)
            
            status_data["last_checked"] = time.time()
            status_data["current_status"] = status
            
            with open(status_file, "w") as f:
                json.dump(status_data, f)
            
            if status == "completed":
                output_file_id = batch_status.output_file_id
                
                # Download results
                print(f"Downloading results for {job_name}...")
                output_content = client.files.content(output_file_id)
                
                # Save raw results
                output_path = os.path.join(batch_dirs["output"], f"{job_name}_output.jsonl")
                with open(output_path, "w") as f:
                    f.write(output_content.text)
                
                # Process and count results
                results = [json.loads(line) for line in output_content.text.strip().split("\n")]
                
                success_count = sum(1 for r in results if r.get("error") is None)
                failure_count = len(results) - success_count
                
                print(f"Processed {success_count} requests successfully, {failure_count} failures")
                
                # Update status file
                with open(status_file, "w") as f:
                    json.dump({
                        "job_name": job_name,
                        "batch_id": batch_id,
                        "status": "completed",
                        "success_count": success_count,
                        "failure_count": failure_count,
                        "completed_at": time.time(),
                        "output_file": output_path
                    }, f)
                
                return True, output_path
                
            elif status in ["failed", "expired", "cancelled"]:
                # Update status file with failure
                with open(status_file, "w") as f:
                    json.dump({
                        "job_name": job_name,
                        "batch_id": batch_id,
                        "status": status,
                        "error": f"Batch {status}",
                        "completed_at": time.time()
                    }, f)
                return False, None
            
            # Wait before checking again
            time.sleep(poll_interval)
            
        except Exception as e:
            print(f"Error checking batch status: {str(e)}")
            # Log error but continue trying
            with open(os.path.join(batch_dirs["logs"], f"{job_name}_error_log.txt"), "a") as f:
                f.write(f"{datetime.datetime.now()}: Error checking status - {str(e)}\n")
    
    # If we get here, it's a timeout
    with open(status_file, "w") as f:
        json.dump({
            "job_name": job_name,
            "batch_id": batch_id,
            "status": "timeout",
            "error": "Timeout waiting for batch completion",
            "completed_at": time.time()
        }, f)
    return False, None

def check_batch_progress(batch_dirs):
    """Check the overall progress of all chunks"""
    status_dir = batch_dirs["status"]
    if not os.path.exists(status_dir):
        return {"total": 0, "completed": 0, "failed": 0, "in_progress": 0}
    
    status_files = [f for f in os.listdir(status_dir) if f.endswith("_status.json")]
    
    total = len(status_files)
    completed = 0
    failed = 0
    in_progress = 0
    
    for status_file in status_files:
        with open(os.path.join(status_dir, status_file), "r") as f:
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

def main():
    args = parse_args()
    if not os.path.exists(args.output_dir): 
        os.makedirs(args.output_dir)
    # Create time-based directories
    timestamp, batch_dirs = create_time_based_dirs(args.output_dir)
    print(f"Created batch run directory for timestamp: {timestamp}")
    
    # Check if input is a file or directory
    jsonl_files = []
    if os.path.isfile(args.jsonl_path):
        # Single file mode
        if args.jsonl_path.endswith('.jsonl'):
            jsonl_files = [args.jsonl_path]
        else:
            print(f"Error: {args.jsonl_path} is not a valid JSONL file")
            return
    elif os.path.isdir(args.jsonl_path):
        # Directory mode - find all JSONL files
        jsonl_files = glob.glob(os.path.join(args.jsonl_path, "*.jsonl"))
        if not jsonl_files:
            print(f"Error: No JSONL files found in directory {args.jsonl_path}")
            return
        print(f"Found {len(jsonl_files)} JSONL files in directory")
    else:
        print(f"Error: {args.jsonl_path} is not a valid file or directory")
        return
    
    # Process each JSONL file
    successful_chunks = 0
    failed_chunks = 0
    output_paths = []
    
    for jsonl_file in jsonl_files:
        print(f"\nProcessing file: {jsonl_file}")
        
        # Split the JSONL file into chunks
        chunk_paths = split_jsonl_into_chunks(jsonl_file, batch_dirs["chunks"], args.chunk_size)
        print(f"Split input file into {len(chunk_paths)} chunks")
        
        # Process each chunk
        for i, chunk_path in enumerate(chunk_paths):
            print(f"\nProcessing chunk {i+1}/{len(chunk_paths)}")
            success, output_path = process_jsonl_batch(
                chunk_path, 
                batch_dirs,
                args.poll_interval,  
                max_retries=args.max_retries
            )
            
            if success and output_path:
                successful_chunks += 1
                output_paths.append(output_path)
            else:
                failed_chunks += 1
            
            # Show overall progress
            progress = check_batch_progress(batch_dirs)
            print(f"Overall progress: {progress['completed']}/{progress['total']} chunks completed, "
                  f"{progress['failed']} failed, {progress['in_progress']} in progress")
    
    # Save final summary
    summary_path = os.path.join(batch_dirs["logs"], "batch_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "batch_dirs": batch_dirs,
            "input_files": jsonl_files,
            "total_chunks_processed": successful_chunks + failed_chunks,
            "successful_chunks": successful_chunks,
            "failed_chunks": failed_chunks,
            "output_paths": output_paths
        }, f, indent=2)
    
    print(f"\nBatch processing complete: {successful_chunks}/{successful_chunks + failed_chunks} chunks processed successfully")
    print(f"Output files are in: {batch_dirs['output']}")
    print(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
