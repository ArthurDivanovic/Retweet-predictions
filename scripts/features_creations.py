
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer
from nltk.corpus import stopwords 
from vaderSentiment_fr.vaderSentiment import SentimentIntensityAnalyzer
from cleaning import cleaning, tokenize
# for the TFID vectorizer
features_vectorizer = 200



"""""""""""""""""""""""""""""""""""""""""""""

Numerical feature creation functions

"""""""""""""""""""""""""""""""""""""""""""""
# convert timestamp to real date
def mois(x):
    return int(datetime.fromtimestamp(x/1000).month)
def jour(x):
    return int(datetime.fromtimestamp(x/1000).day)
def heure(x):
    return int(datetime.fromtimestamp(x/1000).hour)

# create a feature from hashtags based on the frequency of features in all the tweets, the dictionary updates at each call
def feat_tags(data,freq_tags = None):
    tags = []
    lis = []
    cnt = Counter()
    df = data[data['hashtags'] != '[]']
    for hashtag in df['hashtags']:
        words = str(hashtag)
        lis = words[1:-1].split(',')
        for word in lis:
            tags.append(word.replace('\'','').replace(' ',''))
    if (freq_tags != None):
        existing_tags = freq_tags.keys()
        for tag in tags:
            if (tag in existing_tags):
                cnt[tag]+= 1 + freq_tags[tag]
            else :
                cnt[tag]+= 1
    else :
        for tag in tags:
            cnt[tag]+=1
            
    feat = np.zeros(len(data))
    for index, hashtag in enumerate(data['hashtags']):
        words = str(hashtag)
        lis = words[1:-1].split(',')
        for word in lis:
            feat[index] += cnt[word.replace('\'','').replace(' ','')]

    return feat, cnt

def feat_urls(data):
    urls = []
    lis = []
    cnt = Counter()
    df = data[data['urls'] != '[]']
    for url in df['urls']:
        words = str(url)
        lis = words[1:-1].split(',')
        for word in lis:
            urls.append(word.replace('\'','').replace(' ',''))
            
    for url in urls:
        cnt[url]+=1
    feat = np.zeros(len(data))
    feat = np.zeros(len(data))
    for index, url in enumerate(data['urls']):
        words = str(url)
        lis = words[1:-1].split(',')
        for word in lis:
            feat[index] += cnt[word.replace('\'','').replace(' ','')]

    return feat

def tokenize_tags(data):
    words = str(data)
    if (words!= '[]'):
        hashtags = words[1:-1].split(',')
        return [word.replace('\'','').replace(' ','') for word in hashtags]  
    else:
        return []

def tokenize_urls(data):
    words = str(data)
    if (words!= '[]'):
        urls = words[1:-1].split(',')
        return [word.replace('\'','').replace(' ','') for word in urls]  
    else:
        return []

            

def feat_url(data):
    feat = np.zeros(len(data))
    for index, url in enumerate(data['urls']):
        if(url !='[]'):
            feat[index] = 1
    return feat
            
    

# Create sentiment analysis features, based on two methods textblob and vader
def sentiment_analysis(data, method = 'all'):
    if (method == 'textblob') :
        sentiment = data['text'].apply(lambda x : list(TextBlob(x, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer()).sentiment))
        df_sentiments = sentiment.apply(pd.Series)
        df_sentiments.columns = ['polarity', 'subjectivity']
        return df_sentiments
        
    elif (method == 'vader') :
        #Sentiment Analysis
        SIA = SentimentIntensityAnalyzer()
        # Applying Model, Variable Creation
        data['Polarity_vader']=data["text"].apply(lambda x:SIA.polarity_scores(x)['compound'])
        data['Neutral_vader']=data["text"].apply(lambda x:SIA.polarity_scores(x)['neu'])
        data['Negative_vader']=data["text"].apply(lambda x:SIA.polarity_scores(x)['neg'])
        data['Positive_vader']=data["text"].apply(lambda x:SIA.polarity_scores(x)['pos'])      
        return data[['Polarity Score','Neutral Score','Negative Score','Positive Score']]
    
    elif (method =='all'):
        #Sentiment Analysis
        SIA = SentimentIntensityAnalyzer()
        # Applying Model, Variable Creation
        data['Polarity_vader']=data["text"].apply(lambda x:SIA.polarity_scores(x)['compound'])
        data['Neutral_vader']=data["text"].apply(lambda x:SIA.polarity_scores(x)['neu'])
        data['Negative_vader']=data["text"].apply(lambda x:SIA.polarity_scores(x)['neg'])
        data['Positive_vader']=data["text"].apply(lambda x:SIA.polarity_scores(x)['pos'])
        sentiment = data['text'].apply(lambda x : list(TextBlob(x, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer()).sentiment))
        
        df_sentiments = sentiment.apply(pd.Series)
        df_sentiments.columns = ['polarity_blob', 'subjectivity_blob']
        return pd.concat([data[['Polarity_vader','Neutral_vader','Negative_vader','Positive_vader']],df_sentiments],axis=1)
        
    else:
        return None        


      
# The function which gathers all the feature creations
def feat_creation(data,method, freq_tags = None):
    
    features = data.drop(columns = ['text','mentions','urls','hashtags','TweetID'])
    
    features['freq_hashtags'], dic_tags = feat_tags(data, freq_tags)
    features['freq_urls'] = feat_urls(data)
    features['tokenize_tags'] = data['hashtags'].apply(lambda x: tokenize_tags(x))
    features['tokenize_urls'] =  data['urls'].apply(lambda x: tokenize_urls(x))
    features['url'] = feat_url(data)
    features['month'] = data['timestamp'].apply(lambda x: mois(x))
    features['day'] = data['timestamp'].apply(lambda x: jour(x))
    features['hour'] = data['timestamp'].apply(lambda x: heure(x))
    features['nb_urls'] = features['tokenize_urls'].apply(lambda x : int(x.count('\'')/2)  )
    features['nb_tags'] = features['tokenize_tags'].apply(lambda x : int(x.count('\'')/2)  )
    
    df_sentiments = sentiment_analysis(data,method)
    features = pd.concat([features, df_sentiments], axis=1)
    return features, dic_tags
    
# Possibility to normalize the features 
"""
    features_names = features.columns
    features =  MaxAbsScaler().fit_transform(features)
    features = pd.DataFrame(features)
    features.columns = features_names
"""



"""""""""""""""""""""""""""""""""""""""""""""

Functions to transform the text in vectors

"""""""""""""""""""""""""""""""""""""""""""""



# We set up an Tfidf Vectorizer that will use the top 100 tokens from the tweets. We also remove stopwords.
# To do that we have to fit our training dataset and then transform both the training and testing dataset. 
# Here we fit the vectorizer on the most retweeted tweets (first half) from the training dataset
# Then we transform the training data set
def vectorizer_fit(data):
    vectorizer = TfidfVectorizer(max_features=features_vectorizer, stop_words=stopwords.words('french'))   
    vectorizer.fit(data['text'])
    text_data_vect = vectorizer.transform(data['text']).toarray()
    return text_data_vect, vectorizer

# Here we transform the text in a vector based from the vectorizer trained above
def vectorizer_tf_test(data, vectorizer):
    text_data_vect = vectorizer.transform(data['text']).toarray()
    return text_data_vect

# create an embedding dict from pretrained glove on words from tweets to vectors in 200 D
def creation_embedding_dict():
    embed_dict = {}
    with open('embeddings/glove.twitter.27B.200d.txt','r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:],'float32')
            embed_dict[word]=vector
    return embed_dict

# transform tweets in vectors by taking the mean of the vectors of each words from the tweets 
def tweet_to_vec(tweet, embed_dict):
    final_vec = np.zeros(200)
    for word in tweet:
        try:
            final_vec += embed_dict[word]
        except KeyError:
            pass
    return final_vec/len(tweet)

# get all the tweets embbedded
def get_embeds(data, embed_dict):
    clean_data = cleaning(data)
    token_data = tokenize(clean_data)
    text_embeds = token_data['text'].apply(lambda x : tweet_to_vec(x,embed_dict))
    df_text_embeds = text_embeds.apply(pd.Series)
    df_text_embeds.columns = [f'embedding_{i}' for i in range(200)]
    return df_text_embeds




"""""""""""""""""""""""""""""""""""""""""""""

Functions to get train and test datasets

"""""""""""""""""""""""""""""""""""""""""""""
        
# method = ['textblob','vader','all']
# function to get  X_train
def get_X_train_df(dataset,method,embed_dict, freq_tags = None):
    features, dic_tags = feat_creation(dataset,method, freq_tags)
    text, vectorizer = vectorizer_fit(dataset)
    df_text_embeds = get_embeds(dataset, embed_dict)
    vect_text = pd.DataFrame(text,index = features.index)
    vect_text.columns = [f'vector_{i}' for i in range(features_vectorizer)]
    X_final_df = pd.concat([features,vect_text,df_text_embeds], axis=1)
    X_final_df = X_final_df.drop(columns=['retweets_count'])
    return X_final_df, dic_tags, vectorizer

# function to get  X_test
def get_X_test_df(dataset,method, vectorizer,embed_dict, freq_tags):
    features, dic_tags = feat_creation(dataset,method, freq_tags)
    text = vectorizer_tf_test(dataset, vectorizer)
    df_text_embeds = get_embeds(dataset, embed_dict)
    vect_text = pd.DataFrame(text,index = features.index)
    vect_text.columns = [f'vector_{i}' for i in range(features_vectorizer)]
    X_final_df = pd.concat([features,vect_text,df_text_embeds], axis=1)
    return X_final_df, dic_tags



# function to get all the datasets
def get_all_df(X_train, X_test, eval_data, method = 'all'):
    embed_dict = creation_embedding_dict()
    print('Embedding matrix created ! \n')
    X_test = X_test.drop(columns=['retweets_count'])
    X_train = X_train.drop_duplicates()
    X_train, dic_tags, vectorizer = get_X_train_df(X_train,method,embed_dict)
    print('Features created for X_train ! \n')
    X_test, final_freq_tags = get_X_test_df(X_test,method,vectorizer,embed_dict, dic_tags)
    print('Features created for X_test ! \n')
    X_eval, dic = get_X_test_df(eval_data,method,vectorizer,embed_dict, final_freq_tags)
    print('Features created for X_eval ! \n')
    return X_train, X_test, X_eval
    