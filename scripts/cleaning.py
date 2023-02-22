import nltk 
import csv
import numpy as np
import pandas as pd
import html
import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize


# remove all wrong characters from tweets
def cleaning(data):
    clean_data = data.copy()
    #clean duplicates 
    clean_data = clean_data.drop_duplicates(subset=None,keep="first") 
    sw = stopwords.words("french")

    for i, tweet in enumerate(clean_data['text']):
        #remove @
        tweet = re.sub("@[A-Za-z0-9]+","",tweet)
        #remove http links
        tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) 
        tweet = " ".join(tweet.split())
        #Remove hashtag sign but keep the text
        tweet = tweet.replace("#", "").replace("_", " ")
        tweet = tweet.replace("rt","")
        #clean HTML accents 
        tweet = html.unescape(tweet)
        clean_data.iloc[i:i+1,0:1]=tweet 
    return clean_data

# tokenize all the tweets 
def tokenize(data): 
    tokenize_sw_data = data.copy()
    stop_words = set(stopwords.words('french'))
    #tokenize text
    tokenize_sw_data.iloc[:,0] = tokenize_sw_data.iloc[:,0].apply(word_tokenize) 
    
    #remove stop words because stopwords aren't embed in the glove embedding dict    
    #for i, lis in enumerate(tokenize_sw_data['text']):
    #    liste = [word for word in lis if not word.lower() in stop_words]
    #    tokenize_sw_data['text'][i] = liste
        
    return tokenize_sw_data