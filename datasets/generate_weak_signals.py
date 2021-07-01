import numpy as np
import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
from json import JSONDecoder
from functools import partial
import json
from pprint import pprint
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer
import mxnet as mx
# from model_utilities import get_error_bounds






# Preprocessing steps
stemmer = LancasterStemmer()


# to use, download 'glove.42B.300d.txt' from  https://www.kaggle.com/yutanakamura/glove42b300dtxt
# File too large to save to github
df = pd.read_csv('glove.42B.300d.txt', sep=" ", quoting=3, header=None, index_col=0)
glove_model = {key: val.values for key, val in df.T.items()}



# # # # # # # # # #
# Helper Code     #
# # # # # # # # # #


def decodeHTMLencoding(tweets):
    decoded_tweets = tweets.applymap(lambda tweet: BeautifulSoup(tweet, 'lxml').get_text())
    return decoded_tweets

def removeStopWords(text):
    stopw = stopwords.words('english')
    words = [word for word in text.split() if len(word) > 3 and not word in stopw]
    # get stems from words
    for i in range(len(words)):
        words[i] = stemmer.stem(words[i])
    return (" ".join(words)).strip()

def cleanTweets(tweets):
    # decode tweets from html tags
    cleaned_tweets = decodeHTMLencoding(tweets)
    # remove URLs that starts with http
    cleaned_tweets = cleaned_tweets.applymap(lambda tweet: re.sub(
    r'https?:\/\/(www\.)?[-a-zA-Z0–9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%_\+.~#?&//=]*)', '', tweet, flags=re.MULTILINE) )
    # remove URLs that does not start with http
    cleaned_tweets = cleaned_tweets.applymap(lambda tweet: re.sub(
    r'[-a-zA-Z0–9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%_\+.~#?&//=]*)', '', tweet, flags=re.MULTILINE))
    # remove @
    cleaned_tweets = cleaned_tweets.applymap( lambda tweet: re.sub(r'@[A-Za-z0-9_]+', '', tweet, flags=re.MULTILINE) )
    # remove #
    cleaned_tweets = cleaned_tweets.applymap( lambda tweet: re.sub(r'#[A-Za-z0-9_]+', '', tweet, flags=re.MULTILINE) )
    # remove RT
    cleaned_tweets = cleaned_tweets.applymap( lambda tweet: re.sub('RT ', '', tweet, flags=re.MULTILINE) )
    # remove symbols and numbers (i.e keep letters only)
    cleaned_tweets = cleaned_tweets.applymap( lambda tweet: re.sub("[^a-zA-Z]"," ",tweet, flags=re.MULTILINE) )
    #replace consecutive non-ASCII characters with a space
    cleaned_tweets = cleaned_tweets.applymap( lambda tweet: re.sub(r'[^\x00-\x7F]+'," ",tweet.lower(), flags=re.MULTILINE) )
    
    cleaned_tweets.drop_duplicates(inplace=True)
    cleaned_tweets.replace('', np.nan, inplace=True)
    cleaned_tweets.dropna(inplace=True)
    
    return cleaned_tweets

def get_text_vectors(tweets, model):
    # dataset should be a pandas dataframe
    dimension = 300
    data_array = np.empty(shape=[0, dimension])
    indexes = []
    
    for i, tweet in enumerate(tweets):
        words = tweet.split()
        if len(words) !=0:
            feature = 0
            for word in words:
                try:
                    feature += model[word]
                except:
                    pass
            feature /= len(words)
            try:
                if feature.size == dimension:  
                    data_array = np.append(data_array, [feature], axis=0)
                    indexes.append(i)
            except:
                continue
    indexes = np.asarray(indexes)
    assert indexes.size == data_array.shape[0]
    return data_array, indexes



def remove_indices(weak_signals):
    # remove indexes of weak_signals that do not have coverage
    indices = np.where(np.sum(weak_signals, axis=1) == -1*weak_signals.shape[1])[0]
    weak_signals = np.delete(weak_signals, indices, axis=0)
    
    return weak_signals, indices


# test word vectors
# from scipy import spatial
# result = 1 - spatial.distance.cosine(glove_model['horrible'], glove_model['terrible'])
# result

def keyword_labeling(data, keywords, sentiment='pos'):
    mask = 1 if sentiment == 'pos' else 0
    weak_signals = []
    for terms in keywords:
        weak_signal = []
        for text in data:
            label=-1
            for word in terms:
                if word in text.lower():
                    label = mask
            weak_signal.append(label)
        weak_signals.append(weak_signal)
    return np.asarray(weak_signals).T



POSITIVE_LABELS =  [['good','great','nice','delight','wonderful'], 
                    ['love', 'best', 'genuine','well', 'thriller'], 
                    ['clever','enjoy','fine','deliver','fascinating'], 
                    ['super','excellent','charming','pleasure','strong'], 
                    ['fresh','comedy', 'interesting','fun','entertain', 'charm', 'clever'], 
                    ['amazing','romantic','intelligent','classic','stunning'],
                    ['rich','compelling','delicious', 'intriguing','smart']]

NEGATIVE_LABELS = [['bad','better','leave','never','disaster'], 
                   ['nothing','action','fail','suck','difficult'], 
                   ['mess','dull','dumb', 'bland','outrageous'], 
                   ['slow', 'terrible', 'boring', 'insult','weird','damn'],
                   ['drag','awful','waste', 'flat','worse'],
                   #['drag','no','not','awful','waste', 'flat'], 
                   ['horrible','ridiculous','stupid', 'annoying','painful'], 
                   ['poor','pathetic','pointless','offensive','silly']]





# Bellow two functions take from ./cll/model_utilites.py

def calculate_bounds(true_labels, predicted_labels, mask=None):
    """ Calculate error rate on data points the weak signals label """

    if len(true_labels.shape) == 1:
        predicted_labels = predicted_labels.ravel()

    # debugging code
    print("\n\n predicted_labels shape:", predicted_labels.shape)
    print("true_labels shape:", true_labels.shape, "\n\n")

    assert predicted_labels.shape == true_labels.shape

    if mask is None:
        mask = np.ones(predicted_labels.shape)
    if len(true_labels.shape) == 1:
        mask = mask.ravel()

    error_rate = true_labels*(1-predicted_labels) + \
        predicted_labels*(1-true_labels)
    with np.errstate(divide='ignore', invalid='ignore'):
        error_rate = np.sum(error_rate*mask, axis=0) / np.sum(mask, axis=0)
        error_rate = np.nan_to_num(error_rate)
    # check results are scalars
    if np.isscalar(error_rate):
        error_rate = np.asarray([error_rate])
    return error_rate



def get_error_bounds(true_labels, weak_signals):
    """ Get error bounds of the weaks signals
        returns a list of size num_weak x num_classes
    """
    error_bounds = []
    mask = weak_signals >= 0

    for i, weak_probs in enumerate(weak_signals):
        active_mask = mask[i]
        error_rate = calculate_bounds(true_labels, weak_probs, active_mask)
        error_bounds.append(error_rate)
    return error_bounds





# # # # # # # # # # #
# #  SST-2 Dataset  #
# # # # # # # # # # #

def SST_2_generator():
    datapath = './sst-2/'
    train_data = pd.read_csv(datapath+'sst2-train.csv')
    test_data = pd.read_csv(datapath+'sst2-test.csv')
    train_data.head()


    NEGATIVE_LABELS = [['bad','better','leave','never','disaster'], 
                    ['nothing','action','fail','suck','difficult'], 
                    ['mess','dull','dumb', 'bland','outrageous'], 
                    ['slow', 'terrible', 'boring', 'insult','weird','damn'],
                    # ['drag','awful','waste', 'flat','worse'],
                    ['drag','no','not','awful','waste', 'flat'], 
                    ['horrible','ridiculous','stupid', 'annoying','painful'], 
                    ['poor','pathetic','pointless','offensive','silly']]

    positive_labels = keyword_labeling(train_data.sentence.values, POSITIVE_LABELS)
    negative_labels = keyword_labeling(train_data.sentence.values, NEGATIVE_LABELS, sentiment='neg')
    weak_signals = np.hstack([positive_labels, negative_labels])
    weak_signals.shape

    # weak_signals, indices = remove_indices(train_data, weak_signals)

    # debugging code
    print("\n\n Before: \n weak_signals: ",  weak_signals) 
    print("weak_signals shape: ",  weak_signals.shape, "\n\n")



    weak_signals, indices = remove_indices(weak_signals)
    weak_signals.shape

    # debugging code
    print("\n\n after: \n weak_signals: ",  weak_signals) 
    print("weak_signals shape: ",  weak_signals.shape, "\n\n")


    train_labels = train_data.label.values
    test_labels = test_data.label.values

    # n,m = weak_signals.shape
    # weak_signal_probabilities = weak_signals.T.reshape(m,n,1)
    # weak_signals_mask = weak_signal_probabilities >=0
    # true_error_rates = get_error_bounds(train_labels, weak_signal_probabilities, weak_signals_mask)
    # print("error: ", np.asarray(true_error_rates))

    # Clean data and reset index
    train_data.reset_index(drop=True, inplace=True)

    # apply on train data
    train_data = cleanTweets(train_data.drop(columns=['label']))
    # train_data = post_process_tweets(train_data)

    # apply on test data
    test_data = cleanTweets(test_data.drop(columns=['label']))
    # test_data = post_process_tweets(test_data)

    # # debugging code
    # print("\n\n train_data shape: ",  train_data.shape) 
    # print("test_data shape: ",  test_data.shape, "\n\n")

    print(train_data.shape, train_labels.shape)
    print(test_data.shape, test_labels.shape)

    train_features, train_index = get_text_vectors(train_data, glove_model)
    test_features, test_index = get_text_vectors(test_data, glove_model)

    # save sst-2 data
    np.save(datapath+'data_features.npy', train_features)
    np.save(datapath+'test_features.npy', test_features)


    # save sst-2 labels
    np.save(datapath+'data_labels.npy', train_labels[train_index])
    np.save(datapath+'test_labels.npy', test_labels[test_index])

    # # debugging code
    # print("\n\n weak_signals: ",  weak_signals) 
    # print("weak_signals shape: ",  weak_signals.shape)
    # print("\n\ntrain_index: ",  train_index) 
    # print("train_data: ",  train_data) 
    # print("train_data shape: ",  train_data.shape)
    # print("\n\ntest_index: ",  test_index) 
    # print("test_data: ",  test_data) 
    # print("test_data shape: ",  test_data.shape)

    # indexes = train_data
    # indexes = indexes[0]

    m, n = weak_signals.shape 
    indexes = range(0, m)

    # save the one-hot signals
    np.save(datapath+'weak_signals.npy', weak_signals[indexes])


# # # # # # # # #
# IMDB Dataset  # 
# # # # # # # # #

def IMDB_generator():

    datapath = './imdb/'
    df = pd.read_csv(datapath+'IMDB Dataset.csv')

    # apply on train data
    cleaned_data = cleanTweets(df.drop(columns=['sentiment']))
    indexes = cleaned_data.index.values
    df.shape, indexes.size



    n = indexes.size
    # get test data
    np.random.seed(50)
    test_indexes = np.random.choice(indexes, int(n*0.2), replace=False)
    test_labels = np.zeros(test_indexes.size)
    test_labels[df.sentiment.values[test_indexes]=='positive'] = 1
    test_data = df.review.values[test_indexes]

    train_indexes = np.delete(indexes, [np.where(indexes == i)[0][0] for i in test_indexes])
    train_labels = np.zeros(train_indexes.size)
    train_labels[df.sentiment.values[train_indexes]=='positive'] = 1
    train_data = df.review.values[train_indexes]

    print(train_data.shape, train_labels.shape)
    print(test_data.shape, test_labels.shape)


    positive_labels = keyword_labeling(train_data, [['good'],['wonderful'],['great'],['amazing'],['excellent']], sentiment='pos')
    negative_labels = keyword_labeling(train_data, [['bad'],['horrible'],['sucks'],['awful'],['terrible']], sentiment='neg')
    weak_signals = np.hstack([positive_labels, negative_labels])
    weak_signals, indices = remove_indices(weak_signals)
    weak_signals.shape
    
    
    # add signals not covered to test data
    test_data = np.append(test_data, train_data[indices])
    test_labels = np.append(test_labels, train_labels[indices])

    # delete train data not covered by weak signals
    train_data = np.delete(train_data, indices, axis=0)
    train_labels = np.delete(train_labels, indices)

    # get data features
    train_features, train_index = get_text_vectors(train_data, glove_model)
    test_features, test_index = get_text_vectors(test_data, glove_model)

    print(train_index.size, train_data.shape[0])
    test_index.size, test_labels.size


    # save imdb data
    np.save(datapath+'data_features.npy', train_features)
    np.save(datapath+'test_features.npy', test_features)

    # save imdb labels
    np.save(datapath+'data_labels.npy', train_labels[train_index])
    np.save(datapath+'test_labels.npy', test_labels[test_index])


    # debugging code
    print("\n\n weak_signals: ",  weak_signals) 
    print("weak_signals shape: ",  weak_signals.shape)
    print("\n\ntrain_index: ",  train_index) 
    print("train_data: ",  train_data) 
    print("train_data shape: ",  train_data.shape)
    print("\n\ntest_index: ",  test_index) 
    print("test_index: ",  test_data) 
    print("train_data shape: ",  test_data.shape)

    # save the weak_signals
    np.save(datapath+'weak_signals.npy', weak_signals[train_index])


print("\n\n working on SST_2 \n\n" )
SST_2_generator()

# print("\n\n working on IMDB \n\n" )
# IMDB_generator()