import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from data import remove_context_features, remove_std_dvt_context, calc_distance_parameter
from hybridmodel import HybridModel

def readcsv(training_data, data_mode, training_size = None):
    """
    Reads the CSV and outputs the X_train and y_train datasets. Performs data
    scaling if needed.

    :param training_data: training dataset
    :param data_mode: Data scaling to be applied
            0: No data scaling
            1: Standardization
            2: Normalization
    :param training_size: Specifies the percentage of the training_data to be utilized for the training
    of the model
    :return: X_train, y_train
    """

    if isinstance(training_data, pd.DataFrame):
        # X = training_data.drop(columns=[' Delay']) # TODO remove this
        X = training_data.drop(columns=['Drive_cell_size', 'Sink_cell_size'])  # TODO remove this
        X = training_data.iloc[:, 0:training_data.shape[1] - 1]
        y = training_data.iloc[:, training_data.shape[1] - 1]
    else:
        df = pd.read_csv(training_data)
        # df = df.drop(columns=[' Delay'])  # TODO remove this
        df = df.drop(columns=['Drive_cell_size', 'Sink_cell_size'])
        X = df.iloc[:, 0:df.shape[1] - 1]
        y = df.iloc[:, df.shape[1] - 1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.2)

    if training_size is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=1-training_size)

    if (data_mode == 1):  # standardized
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_train = X_train_scaled_df
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

    elif (data_mode == 2):  # normalized
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_train = X_train_scaled_df
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

    return X_train, y_train


def train_method(TRAINING_DATA, TRAINING_SIZE, MAX_TREE_DEPTH, MAX_TREE_FEATURES, LR_type, data_scaling):
    """
    Executes the fit() from HybridModel for training of the model. Called from main.py
    :param TRAINING_DATA: Training dataset
    :param TRAINING_SIZE: Percentage of TRAINING_DATA to be used for training of the model
    :param MAX_TREE_DEPTH: Decision tree parameter, max depth of the tree.
    :param MAX_TREE_FEATURES: Decision tree parameter, max feature usage.
    :param LR_type: Type of linear regressor, 0 for OLS, 1 for Ridge
    :param data_scaling: Type of data scaling to be applied.
            0: No data scaling
            1: Standardization
            2: Normalization
    """
    X_train, y_train = readcsv(TRAINING_DATA, data_scaling, TRAINING_SIZE)

    hb = HybridModel()
    hb.fit(X_train, y_train, LR_type, [MAX_TREE_DEPTH, MAX_TREE_FEATURES])

    with open("hb_instance2.pk1", "wb") as output_file:
        pickle.dump(hb, output_file)
    return


if __name__ == "__main__":
    """
    When the train.py is accesed directly, parameters have to be passed from here
    """
    training_data = "slow.csv"
    training_data = remove_context_features(training_data)
    training_data = remove_std_dvt_context(training_data)
    training_data = calc_distance_parameter(training_data)
                                            #   dataset,       data scaling mode
    X_train, X_test, y_train, y_test = readcsv(training_data, 0)

    hb = HybridModel()   # Linear regressor type, [max_tree_depth, max_tree_feature]
    hb.fit(X_train, y_train, 1, [13, 13])

    with open("hb_instance2.pk1", "wb") as output_file:
        pickle.dump(hb, output_file)

