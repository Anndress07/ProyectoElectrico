import pickle

import pandas as pd
from data import remove_context_features, remove_std_dvt_context, calc_distance_parameter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.metrics import precision_score
FULL_FILES = True
test_data = "designs_slow.csv"
with open("hb_instance2.pk1", "rb") as input_file:
    hb = pickle.load(input_file)


def readcsv_p(training_data, data_mode):

    if isinstance(training_data, pd.DataFrame):
        X = training_data.iloc[:, 0:training_data.shape[1]-1]
        y = training_data.iloc[:, training_data.shape[1]-1 ]
    else:
        df = pd.read_csv(training_data)
        # print(df.columns)
        X = df.iloc[:, 0:df.shape[1] - 1]
        y = df.iloc[:, df.shape[1] - 1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.5)

    if FULL_FILES:
        X_test = X
        y_test = y
        X_train, y_train = None, None
    pd.set_option('display.max_columns', None)
    print(f"desde el predict.py")
    print(f"\ttraining data: {training_data}")
    print(f"dataframe: {df}")
    print(f"\tX_test:\n {X_test}")
    #print(f"y_Test {y_test}")
    #print(f"x_Test: {X_test}")
    # y_test = y_test.head(10000)
    # X_test = X_test.head(10000)
    #y_test = y_test.iloc[[23960, 25870, 56097]]
    #X_test = X_test.iloc[[23960, 25870, 56097]]
    # pd.set_option('display.max_columns', None)
    # print(f"y_Test {y_test}")
    # print(f"x_Test: {X_test}")

    if (data_mode == 1 or data_mode == 2):
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        X_test_scaled = scaler.transform(X_test)
        # X_test_scaled = scaler.fit_transform(X_train)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        X_test = X_test_scaled_df




    return X_train, X_test, y_train, y_test



if __name__ == "__main__":
    new_data = remove_std_dvt_context(test_data)
    new_data = calc_distance_parameter(new_data)
    X_train, X_test, y_train, y_test = readcsv_p(new_data, 0)


    y_lr_pred = hb.predict(X_test)

    with open("hb_instance2.pk1", "wb") as output_file:
        pickle.dump(hb, output_file)
    print("Predict executed directly")



print("se ejecuto el predict")





# TESTING
# # test de data
# for model in hb.leaf_params_dict:
#     if model == 660:
#         nodo_prueba_x = hb.leaf_params_dict[model]
#         print(f"model 660: {hb.leaf_params_dict[660]}")
# for model in hb.leaf_result_dict:
#     if model == 660:
#         nodo_prueba_y = hb.leaf_result_dict[model]
#         print(f"model 660: {hb.leaf_result_dict[660]}")
# nodo_prueba_x = pd.DataFrame(nodo_prueba_x)
# nodo_prueba_x['16'] = nodo_prueba_y
# #print(nodo_prueba_x)
# nodo_prueba_x.to_csv('prueba_lnr.csv', index=False)