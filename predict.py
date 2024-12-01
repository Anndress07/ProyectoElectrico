"""
File: train.py
Description: Executes methods related to the prediction
            of the model. 

Author: Anndress07    
Last update: 1/12/2024

Usage:
            Accessed by main.py
"""

import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from data import remove_context_features, remove_std_dvt_context, calc_distance_parameter




pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)
with open("hb_instance2.pk1", "rb") as input_file:
    hb = pickle.load(input_file)


def readcsv_p(training_data, data_mode, test_size = None):
    """
    Reads the CSV and outputs the X_test and y_test datasets. Performs data
    scaling if needed.

    :param training_data: training dataset
    :param data_mode: Data scaling to be applied
            0: No data scaling
            1: Standardization
            2: Normalization
    :param test_size: Specifies the percentage of the testing_data to be utilized for the testing
    of the model
    :return: X_test, y_test
    """
    if isinstance(training_data, pd.DataFrame):
        design_column = training_data['Design']
        training_data = training_data.drop(columns=['Design'])
        X = training_data.iloc[:, 0:training_data.shape[1]-1]
        y = training_data.iloc[:, training_data.shape[1]-1 ]
    else:
        df = pd.read_csv(training_data)
        # print(f"\nFrom predict.py \n\tDataset w/o modifications is: \n{df}")
        design_column = df['Design']
        df = df.drop(columns=['Design'])
        X = df.iloc[:, 0:df.shape[1] - 1]
        y = df.iloc[:, df.shape[1] - 1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.2)

    if test_size is not None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.9)

    opl_delay_column = X_test[' Delay']     # TODO: remove this
    # X_test = X_test.drop(columns=[' Delay'])
    X_test = X_test.drop(columns=['Drive_cell_size', 'Sink_cell_size'])

    # if the data has scaling
    if (data_mode == 1 or data_mode == 2):
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        X_test = X_test_scaled_df

    design_column.to_csv('design_column.csv', index=False)
    opl_delay_column.to_csv('opl_delay_column.csv', index=False)


    # print(f"\nFrom predict.py")
    # print(f"\tData frame post modifications is: \n{df}")
    # print(f"\tOpenlane Delay column is: \n{opl_delay_column}")

    return X_train, X_test, y_train, y_test



def predict_method(TESTING_DATA, TESTING_SIZE, data_scaling):
    """
    Accesses the predict() method from the HybridModel.

    :param TESTING_DATA: Testing dataset
    :param TESTING_SIZE: Percentage of TESTING_DATA to be used for testing of the model
    :param data_scaling: Type of data scaling to be applied.
            0: No data scaling
            1: Standardization
            2: Normalization
    """
    with open("hb_instance2.pk1", "rb") as input_file:
        hb = pickle.load(input_file)
    X_train, X_test, y_train, y_test = readcsv_p(TESTING_DATA, data_scaling, TESTING_SIZE)

    y_lr_pred = hb.predict(X_test)

    with open("hb_instance2.pk1", "wb") as output_file:
        pickle.dump(hb, output_file)
    # print("Predict executed directly")

    return y_test

if __name__ == "__main__":
    """
        When the predict.py is accesed directly, parameters have to be passed from here
        """
    test_data = "labels_slow.csv"
    test_data = remove_context_features(test_data)
    test_data = remove_std_dvt_context(test_data)
    test_data = calc_distance_parameter(test_data)
    X_train, X_test, y_train, y_test = readcsv_p(test_data, 0)


    y_lr_pred = hb.predict(X_test)

    with open("hb_instance2.pk1", "wb") as output_file:
        pickle.dump(hb, output_file)
    # print("Predict executed directly")


