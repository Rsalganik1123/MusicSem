#!/usr/bin/env python3
import json
import os
import glob
import re
from collections import defaultdict, Counter
import  ast
from tqdm import tqdm
from utils import parse_args


def tuple_to_list(obj):
    """把 tuple 轉成 list，遞迴處理巢狀結構，才能安全轉回 JSON。"""
    if isinstance(obj, tuple):
        return [tuple_to_list(i) for i in obj]
    if isinstance(obj, list):
        return [tuple_to_list(i) for i in obj]
    if isinstance(obj, dict):
        return {k: tuple_to_list(v) for k, v in obj.items()}
    return obj

def clean_snippet(snippet: str) -> str:
    """
    只做『必修』：
    1. 把 ( "foo", "bar" ) → [ "foo", "bar" ]
    2. 移除物件 / 陣列後面的多餘逗號
    3. 把字母 + 裸雙引號 + 字母 換成單引號（修復 T"s / Aren"t）
    其餘不動，保留單引號、註解…留給 ast 處理。
    """
    # 1) tuple → list
    snippet = re.sub(
        r'\(\s*(".*?")\s*,\s*(".*?")\s*\)',
        r'[\1, \2]', snippet
    )
    # 2) trailing comma
    snippet = re.sub(r',\s*(\}|\])', r'\1', snippet)
    # 3) un‑escaped apostrophe
    snippet = re.sub(r'([A-Za-z])"([A-Za-z])', r"\1'\2", snippet)
    return snippet


def extract_json_input_id_to_dict(path):
    extracted, failed = {}, {}

    prefix = os.path.basename(path)
    end_suffix = prefix.find("_batches")
    prefix = prefix[:end_suffix]

    with open(path) as f:
        for line in f:
            data = json.loads(line)
            custom_id = f"{prefix}_{data["custom_id"]}"
            try: 
                body = data["body"]["messages"][0]["content"]
                pos1 = body.find("**Input**")
                if pos1 == -1:
                    continue                                   # 這行沒有 JSON 區塊

                raw = body[pos1 + 9 :].strip()

                fixed = clean_snippet(raw)
                extracted[custom_id] = fixed

            except Exception:
                failed[custom_id] = (raw if 'raw' in locals() else line)
            # break

    return extracted, failed


def extract_json_content(path):
    extracted, failed = [], []

    prefix = os.path.basename(path)
    end_suffix = prefix.find("_batches")
    prefix = prefix[:end_suffix]


    with open(path) as f:
        for line in f:
            try:
                data = json.loads(line)
                custom_id = f"{prefix}_{data["custom_id"]}"
                body = data["response"]["body"]["choices"][0]["message"]["content"]
                pos1 = body.find("```json")
                pos2 = body.find("```", pos1 + 7)
                if pos1 == -1 or pos2 == -1:
                    continue                                   # 這行沒有 JSON 區塊

                raw = body[pos1 + 7 : pos2].strip()
                fixed = clean_snippet(raw)


                # 先試標準 JSON
                try:
                    output_data = json.loads(fixed)
                except json.JSONDecodeError:
                    # 退而求其次：把它當 Python literal 讀進來
                    output_data = ast.literal_eval(fixed)
                    output_data = tuple_to_list(output_data)    
                    
                                 # tuple 轉 list
                output_data["custom_id"] = custom_id
                extracted.append(output_data)

            except Exception:
                output_data = {"raw": (raw if 'raw' in locals() else line)}
                output_data["custom_id"] = custom_id
                failed.append(output_data)

    return extracted, failed


def analyze_song_data(data_list):
    """Analyze the extracted data and calculate statistics."""
    stats = {
        "total_entries": len(data_list),
        "songs": Counter(),
        "category_stats": defaultdict(int),
        "tag_frequency": defaultdict(Counter),
        "empty_categories_count": defaultdict(int),
        "category_tag_counts": defaultdict(int),
        "category_tag_lengths": defaultdict(list),
        "all_tags_per_category": defaultdict(list),  # Changed from set to list to keep all tags
        "category_percentages": defaultdict(float)   # New: Add category percentages
    }
    
    total_tags = 0
    
    for entry in data_list:
        # Process song-artist pairs
        if "pairs" in entry:
            for pair in entry["pairs"]:
                if isinstance(pair, list) and len(pair) == 2:
                    song, artist = pair
                    stats["songs"][f"{song} ({artist})"] += 1
            
            stats["category_stats"]["pairs"] += 1
        
        # Process all categories
        categories = ["Descriptive", "Contextual", "Situational", 
                     "Atmospheric", "Lyrical", "Metadata", "Sentiment"]
        
        for category in categories:
            if category in entry:
                items = entry[category]
                if items:  # If the category has items
                    stats["category_stats"][category] += 1
                    total_tags += len(items)
                    stats["category_tag_counts"][category] += len(items)
                    
                    # Track tag lengths and collect all tags
                    for tag in items:
                        # Handle Ellipsis object (...) specifically
                        if tag is Ellipsis:
                            tag_key = "..."
                        # Convert other unhashable types to strings
                        elif not isinstance(tag, (str, int, float, bool, tuple)):
                            try:
                                tag_key = json.dumps(tag)
                            except TypeError:
                                tag_key = str(tag)
                        else:
                            tag_key = tag
                        
                        stats["tag_frequency"][category][tag_key] += 1
                        
                        # Use string length for non-string tags
                        if isinstance(tag, str):
                            tag_length = len(tag)
                        else:
                            tag_length = len(str(tag))
                        stats["category_tag_lengths"][category].append(tag_length)
                        
                        # Change: Collect all tags, not just unique ones
                        stats["all_tags_per_category"][category].append(tag_key)
                else:
                    stats["empty_categories_count"][category] += 1
            else:
                stats["empty_categories_count"][category] += 1
    
    # Calculate average tags per entry
    stats["avg_tags_per_entry"] = total_tags / len(data_list) if data_list else 0
    
    # Calculate average tag length by category
    stats["avg_tag_length_by_category"] = {}
    for category, lengths in stats["category_tag_lengths"].items():
        if lengths:
            stats["avg_tag_length_by_category"][category] = sum(lengths) / len(lengths)
        else:
            stats["avg_tag_length_by_category"][category] = 0
    
    
    # Count total tags instead of unique
    stats["total_tag_counts"] = {category: len(tags) for category, tags in stats["all_tags_per_category"].items()}
    
    # Calculate category percentages - new addition
    total_entries = len(data_list)
    if total_entries > 0:
        categories = ["Descriptive", "Contextual", "Situational", 
                     "Atmospheric", "Lyrical", "Metadata", "Sentiment", "pairs"]
        for category in categories:
            entries_with_category = stats["category_stats"][category]
            percentage = (entries_with_category / total_entries) * 100
            stats["category_percentages"][category] = round(percentage, 2)
    
    return stats

def extract_thread_name(filename):
    """Extract thread name from filename pattern {thread}_chunk_{id}_output.jsonl"""
    # Fix: Use raw string for regex pattern to avoid escape sequence issues
    pattern = r'([^/]+)_chunk_\d+_output\.jsonl$'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return "unknown"


def convert_for_json(obj):
    """Convert sets and other non-JSON-serializable objects to serializable types."""
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_for_json(item) for item in obj]
    elif obj is Ellipsis:  # Handle Ellipsis object
        return "..."
    return obj

def process_directory(input_batch_path, output_batch_path, output_folder="output_data"):
    """Process all JSONL files in the directory, grouping by thread."""
    thread_data = defaultdict(list)
    thread_failed = defaultdict(list)
    file_count = 0
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all JSONL files in the directory
    # Fix: Use raw string for regex pattern
    input_jsonl_files = glob.glob(os.path.join(input_batch_path, "**", "*.jsonl"), recursive=True) 

    input_dict = {}
    print("get input dict...")
    for jsonl_file in tqdm(input_jsonl_files):
        success, _ = extract_json_input_id_to_dict(jsonl_file)
        input_dict.update(success)


    jsonl_files = glob.glob(os.path.join(output_batch_path, "**", "*.jsonl"), recursive=True)
    
    for jsonl_file in jsonl_files:
        thread_name = extract_thread_name(jsonl_file)
        data, failed = extract_json_content(jsonl_file)
        for line in data:
            line["input"] = input_dict[line["custom_id"]]
        for line in failed:
            line["input"] = input_dict[line["custom_id"]]
        thread_data[thread_name].extend(data)
        thread_failed[thread_name].extend(failed)
        file_count += 1
        print(f"Processed {jsonl_file}: {len(data)} entries for thread {thread_name}")
    
    # Analyze each thread separately
    thread_stats = {}
    thread_averages = {}  # New: separate averages
    thread_percentages = {}  # New: category percentages by thread
    all_data = []
    all_failed = []
    
    for thread_name, data in thread_data.items():
        stats = analyze_song_data(data)
        # Extract averages to separate structure
        thread_averages[thread_name] = {
            "avg_tags_per_entry": stats["avg_tags_per_entry"],
            "avg_tag_length_by_category": stats["avg_tag_length_by_category"]
        }
        # Extract percentages to separate structure
        thread_percentages[thread_name] = stats["category_percentages"]
        # Remove averages and percentages from main stats
        stats.pop("avg_tags_per_entry")
        stats.pop("avg_tag_length_by_category")
        stats.pop("category_percentages")
        thread_stats[thread_name] = stats
        all_data.extend(data)
        all_failed.extend(thread_failed[thread_name])
    
    # Overall stats, averages, and percentages
    overall_stats = analyze_song_data(all_data)
    overall_averages = {
        "avg_tags_per_entry": overall_stats["avg_tags_per_entry"],
        "avg_tag_length_by_category": overall_stats["avg_tag_length_by_category"]
    }
    overall_percentages = overall_stats["category_percentages"]
    
    overall_stats.pop("avg_tags_per_entry")
    overall_stats.pop("avg_tag_length_by_category")
    overall_stats.pop("category_percentages")
    overall_stats["file_count"] = file_count
    
    # Add averages to separate structure
    averages_result = {
        "by_thread": thread_averages,
        "overall": overall_averages
    }
    
    # Add percentages to separate structure
    percentages_result = {
        "by_thread": thread_percentages,
        "overall": overall_percentages
    }
    
    # Save all data to files - convert non-serializable objects first
    with open(os.path.join(output_folder, "extracted_data.json"), "w", encoding="utf-8") as f:
        json.dump(convert_for_json(thread_data), f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(output_folder, "failed_extractions.json"), "w", encoding="utf-8") as f:
        json.dump(convert_for_json(thread_failed), f, indent=2, ensure_ascii=False)
    
    stats_result = {
        "by_thread": thread_stats,
        "overall": overall_stats
    }
    
    # Save main stats
    with open(os.path.join(output_folder, "song_analysis_stats.json"), "w", encoding="utf-8") as f:
        json.dump(convert_for_json(stats_result), f, indent=2, ensure_ascii=False)
    
    # Save averages to new file
    with open(os.path.join(output_folder, "song_analysis_averages.jsonl"), "w", encoding="utf-8") as f:
        for thread_name, avg_data in averages_result["by_thread"].items():
            f.write(json.dumps({"thread": thread_name, "averages": avg_data}, ensure_ascii=False) + "\n")
        f.write(json.dumps({"thread": "overall", "averages": averages_result["overall"]}, ensure_ascii=False) + "\n")
    
    # Save percentages to new file - similar to averages file
    with open(os.path.join(output_folder, "category_percentages.jsonl"), "w", encoding="utf-8") as f:
        for thread_name, percentage_data in percentages_result["by_thread"].items():
            f.write(json.dumps({"thread": thread_name, "percentages": percentage_data}, ensure_ascii=False) + "\n")
        f.write(json.dumps({"thread": "overall", "percentages": percentages_result["overall"]}, ensure_ascii=False) + "\n")
    
    return stats_result, averages_result, percentages_result

def main(input_dir):
    """Main function to run the analysis."""
    # Default path - update as needed
    batch_output_path = f"{input_dir}/output"
    batch_intput_path = f"{input_dir}/chunks"
    # batch_output_path = "/data2/rsalgani/reddit/jsonl_extract_batch/batch_run_20250426_222151/output"
    # batch_intput_path = "/data2/rsalgani/reddit/jsonl_extract_batch/batch_run_20250426_222151/chunks"

    
    # Create a timestamped output folder
    timestamp = os.path.basename(batch_output_path.rstrip("/"))
    output_folder = f"{input_dir}/processed_extractions_{timestamp}"
    
    # Process all JSONL files in the directory
    stats_result, averages_result, percentages_result = process_directory(batch_intput_path, batch_output_path, output_folder)

    # Print summary
    overall = stats_result["overall"]
    threads = stats_result["by_thread"]
    print(f"Analysis complete. Results saved to {output_folder}/")
    print(f"Overall: Processed {overall['file_count']} files with {overall['total_entries']} entries.")
    for thread, thread_stats in threads.items():
        print(f"Thread '{thread}': {thread_stats['total_entries']} entries.")

if __name__ == "__main__":
    args = parse_args()
    main(args.input_dir)
