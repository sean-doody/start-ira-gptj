import os
import time
import glob
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import TweetTokenizer
from joblib import Parallel, delayed
tqdm.pandas()


def make_data(data: list):
    cols = ["tweetid", 
            "user_screen_name", 
            "tweet_language", 
            "tweet_text", 
            "tweet_time", 
            "is_retweet", 
            "in_reply_to_tweetid", 
            "hashtags", 
            "urls", 
            "user_mentions"]
    
    df = pd.DataFrame()
    
    for d in data:
        print(f"LOADING {d}")
        for chunk in pd.read_csv(d, chunksize=250000, low_memory=False, usecols=cols):
            df = pd.concat([df, chunk])
            
    # drop any possible duplicates:
    og_len = len(df)
    df.drop_duplicates(subset="tweetid", inplace=True)     
    deduped = len(df)
    
    print(f"Found {og_len - deduped} duplicates")
    
    # extract english tweets:
    print("Extracting English tweets")
    df = df[df.tweet_language == "en"]
    
    # remove retweets:
    df = df[df.is_retweet == False]
    
    # remove excessively short tweets:
    print("Counting tokens")
    tokenizer = TweetTokenizer()
    df["tokens"] = Parallel(n_jobs=7)(delayed(tokenizer.tokenize)(line) for line in tqdm(df.tweet_text))
    df["n_tokens"] = df.tokens.apply(lambda row: len(row))
    df = df[df.n_tokens >= 10]
    
    print(f"Final sample size: {len(df)}")
    
    # save:
    print("Saving")
    df.reset_index(inplace=True, drop=True)
    df.to_feather(os.path.join("raw-twitter", "processed", "ru_en_tweets.feather"))


if __name__ == "__main__":
    start = time.time()
    
    files = glob.glob(os.path.join("raw-twitter", "*.csv"))
    make_data(files)
    
    end = time.time()
    timer = round((end - start) / 60, 3)
    
    print("Done!")
    print(f"Run time: {timer} minutes")    