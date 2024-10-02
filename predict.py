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
with open("hb_instance.pk1", "rb") as input_file:
    hb = pickle.load(input_file)


def readcsv(training_data):
    df = pd.read_csv(training_data)
    X = df.iloc[:, 0:16]
    y = df.iloc[:, 16]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.5)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = readcsv(test_data)
y_lr_pred = hb.predict(X_test)

with open("hb_instance.pk1", "wb") as output_file:
    pickle.dump(hb, output_file)
