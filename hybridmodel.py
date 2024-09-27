import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error


class HybridModel:

    def __init__(self):
        return


    def tree(self, training_data):
        """
        Builds the initial decision tree used for sample classification
        :param training_data: dataset to train the tree
        :return dtr: the decision tree object
        :X_test: test dataset
        y_test: test output variable
        """

        df = pd.read_csv(training_data)
        X = df.iloc[:, 0:16]
        y = df.iloc[:, 16]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.5)

        dtr = DecisionTreeRegressor(max_depth=9, max_features=15, random_state=10)

        dtr.fit(X_train, y_train)

        y_pred = dtr.predict(X_test)
        y_pred_train = dtr.predict(X_train)

        return dtr, X_test, y_test

    def classification(self, dtr, X_test, y_test):
        """
        Classifies each sample depending on the leaf node of the decision tree they land in
        using the apply() method. Creates a dictionary where each node contains all its samples in a list
            leaf_params_dict = {ID1: [ [sample1],[sample2]...], ID2: [ [sample3],[sample4]...]... }
            leaf_result_dict = {ID1: [ [y1, y2]...]. ID2: [y3, y4]...}
        :param dtr: decision tree structure
        :param X_test: parameter training set
        :param y_test: prediction set
        :return leaf_params_dict: Contains all leaf nodes with its grouped sample parameters:
        :return leaf_result_dict: Contains all leaf nodes with its grouped sample predictions
        """
        leaf_sample_list = dtr.apply(X_test)

        leaf_params_dict = {}
        leaf_result_dict = {}

        for leaf_node in range(len(X_test)):
            leaf_id_value = leaf_sample_list[leaf_node]
            if leaf_id_value not in leaf_params_dict:
                leaf_params_dict[leaf_id_value] = []
            if leaf_id_value not in leaf_result_dict:
                leaf_result_dict[leaf_id_value] = []

            leaf_params_dict[leaf_id_value].append(X_test.iloc[leaf_node].tolist())
            leaf_result_dict[leaf_id_value].append(y_test.iloc[leaf_node])

        # nodo_prueba_x = []
        # for leaf_node in range(len(X_test)):
        #     if leaf_sample_list[leaf_node] == 877:
        #         nodo_prueba_x.append(X_test.iloc[leaf_node].tolist())
        #         nodo_prueba_x.append(y_test.iloc[leaf_node].tolist())
        # dfx = pd.DataFrame(nodo_prueba_x)
        # dfx.to_csv("max_error.csv", index=False)
        return leaf_sample_list, leaf_params_dict, leaf_result_dict