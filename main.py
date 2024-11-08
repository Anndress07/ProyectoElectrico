import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from sklearn.model_selection import train_test_split
from sklearn import tree
from data import remove_context_features_two, remove_std_dvt_context_two, calc_distance_parameter_two
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from train import train_method
from predict import predict_method
from results_run import results_method

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.metrics import precision_score

TRAINING_DATA = 'slow.csv'
TRAINING_SIZE = 0.8
MAX_TREE_DEPTH = 13
MAX_TREE_FEATURES = 13


TESTING_DATA = 'labels_slow.csv'
TESTING_SIZE = 1.0

context_features = False
std_dvt_context = False
distance_parameter = True

LR_type = 1
data_scaling = 0

plots_enable = True

def main():

    """     TRAINING        """
    train_method(TRAINING_DATA, TRAINING_SIZE, MAX_TREE_DEPTH, MAX_TREE_FEATURES, LR_type, data_scaling)
    print(f"at training: {TRAINING_DATA}")

    """     PREDICTING      """
    y_test = predict_method(TESTING_DATA, TESTING_SIZE, data_scaling)
    print(f"at predicting: {TESTING_DATA}")

    """     RESULTS DISPLAY """
    results_method(y_test, plots_enable)

    return

def feature_modding():
    global TRAINING_DATA
    global TESTING_DATA
    if not context_features:
        TRAINING_DATA, TESTING_DATA, TESTING_DATA = remove_context_features_two(TRAINING_DATA,TESTING_DATA, TESTING_DATA)
    if not std_dvt_context:
        TRAINING_DATA, TESTING_DATA, TESTING_DATA = remove_std_dvt_context_two(TRAINING_DATA, TESTING_DATA, TESTING_DATA)
    if distance_parameter:
        TRAINING_DATA, TESTING_DATA, TESTING_DATA = calc_distance_parameter_two(TRAINING_DATA, TESTING_DATA, TESTING_DATA)
    return




if __name__ == "__main__":
    feature_modding()
    main()
