import numpy as np
from train_classifier import *
from ge_criterion_baseline import *
from utilities import saveToFile, runBaselineTests, getModelAccuracy, getWeakSignalAccuracy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import default_reader
from real_experiments import run_experiment, bound_experiment, dependent_error_exp


def run_tests():
    """
    Runs experiment.
    :return: None
    """
 
    # # # # # # # # # # # #
    # breast cancer       #
    # # # # # # # # # # # #

    #for breast cancer classification dataset, select the mean radius, radius se and worst radius as weak signals
    print("\n\n# # # # # # # # # # # # # # # # # # # # #")
    print("# Running breast cancer experiment...   #")
    print("# # # # # # # # # # # # # # # # # # # # #\n")
    views                  = {0:0, 1:10, 2:20}
    datapath               = 'datasets/breast-cancer/wdbc.data'
    savepath               = 'results/json/breast_cancer.json'
    load_and_process_data  = default_reader.breast_cancer_load_and_process_data
    default_reader.run_experiment(run_experiment, saveToFile, views, datapath, load_and_process_data, savepath)




    # # # # # # # # # # # #
    # obs network         #
    # # # # # # # # # # # #

    # #for obs network dataset, select the Utilized Bandwidth Rate, Packet drop rate and Flood Status as weak signals
    print("\n\n\n\n# # # # # # # # # # # # # # # # # # # # #")
    print("# Running obs network experiment...     #")
    print("# # # # # # # # # # # # # # # # # # # # #\n")
    views                  = {0:1, 1:2, 2:20}
    datapath               = 'datasets/obs-network/obs_network.data'
    savepath               = 'results/json/obs_network.json'
    load_and_process_data  = default_reader.obs_load_and_process_data
    default_reader.run_experiment(run_experiment, saveToFile, views, datapath, load_and_process_data, savepath)



    # # # # # # # # # # # #
    # cardio              #
    # # # # # # # # # # # #
 
    # #Use AC, MLTV and Median as weak signal views
    print("\n\n\n\n# # # # # # # # # # # # # # # # # # # # #")
    print("# Running cardio experiment...          #")
    print("# # # # # # # # # # # # # # # # # # # # #\n")
    views                  = {0:1, 1:10, 2:18}
    datapath               = 'datasets/cardiotocography/cardio.csv'
    savepath               = 'results/json/cardio.json'
    load_and_process_data  = default_reader.cardio_load_and_process_data
    default_reader.run_experiment(run_experiment, saveToFile, views, datapath, load_and_process_data, savepath)


def run_bounds_experiment():
    """
    Runs experiment.
    :return: None
    """

    # # # # # # # # # # # #
    # breast cancer       #
    # # # # # # # # # # # #
    
    # for breast cancer classification dataset, select the mean radius, radius se and worst radius as weak signals
    views                     = {0:0, 1:10, 2:20}
    path                      = 'results/json/bc_bounds.json'
    load_and_process_data     = default_reader.breast_cancer_load_and_process_data
    data_and_weak_signal_data = default_reader.create_weak_signal_view('datasets/breast-cancer/wdbc.data', views, load_and_process_data)
    default_reader.run_bounds_experiment(bound_experiment, data_and_weak_signal_data, path) 

    # # # # # # # # # # # #
    # obs network         #
    # # # # # # # # # # # #

    # for obs network dataset, select the Utilized Bandwidth Rate, Packet drop rate and Flood Status as weak signals
    views                     = {0:1, 1:2, 2:20}
    path                      = 'results/json/obs_bounds.json'
    load_and_process_data     = default_reader.obs_load_and_process_data
    data_and_weak_signal_data = default_reader.create_weak_signal_view('datasets/obs-network/obs_network.data', views, load_and_process_data)
    default_reader.run_bounds_experiment(bound_experiment, data_and_weak_signal_data, path)
    obs_network_reader.run_bounds_experiment(bound_experiment)




def run_dep_err_experiment():

    # # # # # # # # # # # #
    # cardio              #
    # # # # # # # # # # # #

    print("Running dependent error on cardio experiment...")
     
    #Use AC, MLTV and Median as weak signal views
    views = {0:1, 1:18}
    # repeat the bad weak signal 
    for i in range(2,10):
        views[i] = 18
    path                      = 'results/json/cardio_error.json'
    load_and_process_data     = default_reader.cardio_load_and_process_data
    data_and_weak_signal_data = default_reader.create_weak_signal_view('datasets/cardiotocography/cardio.csv', views, load_and_process_data)
    default_reader.run_dep_error_exp(dependent_error_exp, data_and_weak_signal_data, path)

if __name__ == '__main__':
    run_tests()

    # # un-comment to run bounds experimrnt in the paper
    # run_bounds_experiment()

    # # un-comment to run dependency error experiment in the paper
    #run_dep_err_experiment()