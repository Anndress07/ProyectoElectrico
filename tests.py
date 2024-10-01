from hybridmodel import HybridModel
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


df = pd.read_csv("slow.csv")
pd.set_option('display.max_columns', None)
X = df.iloc[:, 0:16]
y = df.iloc[:, 16]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.5)

hb = HybridModel()
decision_tree_object, param_dict, output_dict, LR_results =  hb.fit(X_train, y_train)
#y_lr_pred = hb.predict(X_test)
with open("hb_instance.pk1", "wb") as output_file:
    pickle.dump(hb, output_file)





# print(y_test)
# print(y_lr_pred)
#
#
# print("--tree")
# print("MAE test", mean_absolute_error(y_test, y_pred))
# print("Mean Squared Error (MSE) test:", r2_score(y_test, y_pred))
# print("R-squared Score test: ", mean_squared_error(y_test, y_pred))
#
# print("--linear reg")
# print("MAE test", mean_absolute_error(y_test, y_lr_pred))
# print("Mean Squared Error (MSE) test:", r2_score(y_test, y_lr_pred))
# print("R-squared Score test: ", mean_squared_error(y_test, y_lr_pred))


