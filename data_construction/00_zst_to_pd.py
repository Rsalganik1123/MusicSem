import ipdb 
import zstandard as zstd
import pandas as pd 
import ujson as json
import pickle 
from utils import Zreader, parse_args


def load_chunks(thread:str, fp:str, mode:str): 
    """
    Loads the reddit data from .zst format into pandas dataframe

    Args:
        thread (str): Name of the thread.
        fp (str): File path leading to reddit database folder.
        mode (str): Specifying whether you are reading submissions (posts) or comments 

    Returns:
        pandas dataframe: submissions or comments.
    """
    # Adjust chunk_size as necessary -- defaults to 16,384 if not specified
    reader = Zreader(f"{fp}raw_data/{thread}_{mode}.zst", chunk_size=8192)
    
    if mode == 'submissions': 
        keys = ['id', 'subreddit_id', 'selftext', 'url', 'author', 'title', 'created_utc', 'num_comments']
    if mode == 'comments': 
        keys = ['link_id', 'parent_id','id', 'subreddit_id', 'body',  'author', 'created_utc']
    all_data = [] 
    # Read each line from the reader
    for line in reader.readlines():
        try: 
            obj = json.loads(line)
            data = {key: value for key, value in obj.items() if key in keys}
            # print(data)
            if mode == 'submissions': 
                if data['num_comments'] > 1 or len(data['selftext']) > 1: 
                    all_data.append(data)
            if mode == 'comments': 
                if len(data['body']) > 1: 
                    data['link_id'] = data['link_id'][3:]
                    all_data.append(data)
        except: 
            print('read failed, skipping')
    df = pd.DataFrame(all_data)
    print(f"loaded {len(df)} {mode}")
    pickle.dump(df, open(f'{fp}filtered_data/{thread}_{mode}.pkl', 'wb'))
    
def join_posts_and_comments(thread:str, fp:str):
    """
    Joins the posts and comments

    Args:
        thread (str): Name of the thread.
        fp (str): File path leading to reddit database folder.

    Returns:
        pandas dataframe: joined between submissions and comments.
    """
    subs = pickle.load(open(f'{fp}filtered_data/{thread}_submissions.pkl', 'rb'))
    comms = pickle.load(open(f'{fp}filtered_data/{thread}_comments.pkl', 'rb'))
    joined = pd.merge(subs, comms, left_on='id', right_on='link_id')
    no_moderator = joined[joined.author_y != 'AutoModerator']
    no_deleted = no_moderator[no_moderator.body != '[deleted]']
    no_removed = no_deleted[no_deleted.body != '[removed]']
    no_removed = no_removed.reset_index(drop=True)
    print(f"loaded {len(no_removed)} entries")
    pickle.dump(no_removed, open(f'{fp}joined_data/{thread}_joined.pkl', 'wb'))

if __name__ == '__main__': 
    """
    Note: This code will load from .zst --> individual pandas dataframe --> joint dataframe
            Join performed on basis of post_id
    Returns:
        pandas dataframe: joined between submissions and comments.
    """
    args = parse_args()
    load_chunks(args.thread, args.fp, mode = 'comments')
    load_chunks(args.thread, args.fp, mode = 'submissions') 
    join_posts_and_comments(args.thread, args.fp)
    
