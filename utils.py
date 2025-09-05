import re
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks    
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    # remove hashtags
    tweet = re.sub(r'#', '', tweet)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  
                word not in string.punctuation):  
            stem_word = stemmer.stem(word) 
            tweets_clean.append(stem_word)

    return tweets_clean


def build_freqs(tweets,ys):
    yslist = np.squeeze(ys).tolist()
    freqs = {}
    for y,tweet in zip(yslist,tweets):
        for word in process_tweet(tweet):
            pair = (word,y)
            if pair in freqs:
                freqs[pair]+=1
            else:
                freqs[pair] = 1
    return freqs

def extract_features(processed_tweet,freqs):
    # [bias,positive_freqs,negative_freqs]
    vec = np.zeros((1,3))
    
    #Bias term would be one
    vec[0,0] = 1
    
    for word in processed_tweet:
        pos_freq = freqs.get((word,1),0)
        vec[0,1] += pos_freq
        neg_freq = freqs.get((word,0),0)
        vec[0,2] += neg_freq
    return vec


    