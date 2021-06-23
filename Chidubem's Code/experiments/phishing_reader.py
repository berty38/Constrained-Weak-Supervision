import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from experiments import default_reader

def load_and_process_data(path):

    """Loads text classification data from `path`"""
    df = pd.read_csv(path, header=None)

    #replace labels 1 with 0 and -1 with 1
    df[30] = df[30].replace({1:0, -1:1})
    data_matrix = df.values

    #Split the data into 70% training and 30% test set
    data_labels = data_matrix[:,-1:].ravel() 
    data_matrix = data_matrix[:,:-1]
    train_data, test_data, train_labels, test_labels = train_test_split(data_matrix, data_labels.astype('float'), test_size=0.3, shuffle=True, stratify=data_labels)

    #Normalize the features of the data
    scaler = preprocessing.StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    assert train_labels.size == train_data.shape[0]
    assert test_labels.size == test_data.shape[0]

    data = {}

    val_data, weak_supervision_data, val_labels, weak_supervision_labels = train_test_split(train_data, train_labels.astype('float'), test_size=0.4285, shuffle=True, stratify=train_labels)

    data['training_data'] = weak_supervision_data, weak_supervision_labels
    data['validation_data'] = val_data, val_labels
    data['test_data'] = test_data, test_labels

    return data


def run_experiment(run, save):

    """
    :param run: method that runs real experiment given data
    :type: function
    :param save: method that saves experiment results to JSON file
    :type: function
    :return: none
    """

    #Use url of anchor, web traffic and Google index as weak signal views
    views = {0:13, 1:25, 2:27}
    datapath = 'datasets/phishing/phishing.txt'
    savepath = 'results/json/phishing.json'
    default_reader.run_experiment(run, save, views, datapath, load_and_process_data, savepath)


def run_bounds_experiment(run):

    """
    :param run: method that runs real experiment given data
    :type: function
    :return: none
    """
    #Use url of anchor, web traffic and Google index as weak signal views
    views = {0:13, 1:25, 2:27}
    path = 'results/json/phishing_bounds.json'
    data_and_weak_signal_data = default_reader.create_weak_signal_view('datasets/phishing/phishing.txt', views, load_and_process_data)
    default_reader.run_bounds_experiment(run, data_and_weak_signal_data, path)


def run_dep_error_exp(run):

    """
    :param run: method that runs real experiment given data
    :type: function
    :return: none
    """
    #Use url of anchor, web traffic and Google index as weak signal views
    views = {0:25, 1:27}
    # repeat the bad weak signal 
    for i in range(2,10):
        views[i] = 27
    path = 'results/json/phishing_error.json'
    data_and_weak_signal_data = default_reader.create_weak_signal_view('datasets/phishing/phishing.txt', views, load_and_process_data)
    default_reader.run_dep_error_exp(run, data_and_weak_signal_data, path)