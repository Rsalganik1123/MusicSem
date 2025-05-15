import ipdb 
import zstandard as zstd
import pandas as pd 
import ujson as json
import argparse

MODEL = "gpt-4o"
POLL_INTERVAL = 120  # seconds between status checks
CHUNK_SIZE = 10000  # Requests per chunk


def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='/data2/rsalgani/reddit/joined_data/', help='path to the input folder here')  
    parser.add_argument('--output_dir', help='path to output folder')
    parser.add_argument('--audio_dir', help='path to audio folder')
    parser.add_argument('--gathered_file')
    parser.add_argument('--thread', help='this is going to be the thread name (e.g. LetsTalkMusic)')    
    parser.add_argument("--jsonl_path", default="test", 
                       help="Path to a JSONL file or directory containing JSONL files")
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE, 
                        help=f"Number of requests per chunk (default: {CHUNK_SIZE})")
    parser.add_argument("--poll_interval", type=int, default=POLL_INTERVAL, 
                        help=f"Interval in seconds between status checks (default: {POLL_INTERVAL})")
    parser.add_argument("--max_retries", type=int, default=100,
                        help="Maximum number of status check retries (default: 100)")
    args = parser.parse_args()

    return args

class Zreader:

    def __init__(self, file, chunk_size=16384):
        '''Init method'''
        self.fh = open(file,'rb')
        self.chunk_size = chunk_size
        self.dctx = zstd.ZstdDecompressor()
        self.reader = self.dctx.stream_reader(self.fh)
        self.buffer = ''


    def readlines(self):
        '''Generator method that creates an iterator for each line of JSON'''
        while True:
            try: 
                chunk = self.reader.read(self.chunk_size).decode()
            except: 
                print('read failed, skipping')
                continue 
            if not chunk:
                break
            
            lines = (self.buffer + chunk).split("\n")
            

            for line in lines[:-1]:
                yield line

            self.buffer = lines[-1]

