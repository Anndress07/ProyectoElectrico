from hybridmodel import HybridModel
from data import remove_context_features, remove_std_dvt_context, calc_distance_parameter
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.metrics import precision_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler
FULL_FILES = True # if the entire file is used to train
training_data = "slow.csv"
# with open("hb_instance2.pk1", "rb") as input_file:
#     hb = pickle.load(input_file)
def readcsv(training_data, data_mode):
    if isinstance(training_data, pd.DataFrame):
        X = training_data.iloc[:, 0:training_data.shape[1]-1]
        y = training_data.iloc[:, training_data.shape[1]-1]
    else:
        df = pd.read_csv(training_data)
        # print(df.columns)
        # X = df.iloc[:, 0:16]
        # y = df.iloc[:, 16]
        X = df.iloc[:, 0:df.shape[1] - 1]
        y = df.iloc[:, df.shape[1] - 1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.5)
    if FULL_FILES:
        X_train = X
        y_train = y
        X_test, y_test = None, None

    if (data_mode == 1): #estandarizada
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_train = X_train_scaled_df
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

    elif (data_mode == 2): #normalizada
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_train = X_train_scaled_df
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

    return X_train, X_test, y_train, y_test
# X_train, X_test, y_train, y_test = readcsv(training_data)

if __name__ == "__main__":
    new_data = remove_std_dvt_context(training_data)
    new_data = calc_distance_parameter(new_data)
    X_train, X_test, y_train, y_test = readcsv(new_data, 0)

    hb = HybridModel()
    hb.fit(X_train, y_train, 1)

    with open("hb_instance2.pk1", "wb") as output_file:
        pickle.dump(hb, output_file)
    print("Predict executed directly")

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)









# if __name__ == "__main__":
#     X_train, X_test, y_train, y_test = readcsv(training_data, 0)
#
#     hb = HybridModel()
#     hb.fit(X_train, y_train)
#
#     with open("hb_instance2.pk1", "wb") as output_file:
#         pickle.dump(hb, output_file)
#     print("Train executed directly")