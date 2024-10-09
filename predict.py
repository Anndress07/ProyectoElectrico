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

test_data = "slow.csv"
with open("hb_instance2.pk1", "rb") as input_file:
    hb = pickle.load(input_file)


def readcsv(training_data):
    df = pd.read_csv(training_data)
    X = df.iloc[:, 0:16]
    y = df.iloc[:, 16]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.5)
    #print(f"y_Test {y_test}")
    #print(f"x_Test: {X_test}")
    # y_test = y_test.head(10000)
    # X_test = X_test.head(10000)
    pd.set_option('display.max_columns', None)
    # print(f"y_Test {y_test}")
    # print(f"x_Test: {X_test}")
    return X_train, X_test, y_train, y_test

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

X_train, X_test, y_train, y_test = readcsv(test_data)

X_test = scaler.transform(X_test)
y_lr_pred = hb.predict(X_test)

# TESTING
with open("hb_instance2.pk1", "wb") as output_file:
    pickle.dump(hb, output_file)
# # test de data
# for model in hb.leaf_params_dict:
#     if model == 423:
#         nodo_prueba_x = hb.leaf_params_dict[model]
#         print(f"model 192: {hb.leaf_params_dict[192]}")
# for model in hb.leaf_result_dict:
#     if model == 423:
#         nodo_prueba_y = hb.leaf_result_dict[model]
#         print(f"model 192: {hb.leaf_result_dict[192]}")
# nodo_prueba_x = pd.DataFrame(nodo_prueba_x)
# nodo_prueba_x['16'] = nodo_prueba_y
# print(nodo_prueba_x)
# nodo_prueba_x.to_csv('prueba_lnr.csv', index=False)
