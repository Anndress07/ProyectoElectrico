import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error


class HybridModel:
    """
    Contains the structure for the entire hybrid model, the decision tree and each linear regression for the leaf nodes,
    as well as the methods necessary for training ( fit() ) and prediction ( predict() )
    The fit() method is meant to be used first with the training data as parameters (both X_train and y_train). It will
    generate all objects needed for the training.
    After calling fit(), predict() can be used with the X_test (or the data to predict). It will yield
    linear_predictions, a list with the output for each sample in X_test.

    """

    def __init__(self):
        print("Running the hybrid class")
        return

    def fit(self, X_train, y_train):
        """
        Trains the hybrid model, first creating the decision tree object using the tree() method. With the method
        classification() each sample is classified depending on the landing leaf node in the decision tree. Finally,
        the linear regression is performed for each leaf node. The relevant coefficients are saved in a dicitonary and
        used for later prediction methods.
        :param X_train: training dataset parameters
        :param y_train: training dataset outputs
        :return decision_tree_object: the decision tree object generated in tree()
        :return param_dict: dictionary with all samples classified by leaf node ID
        :return LR_results: dictionary with all linear regression coefficients classified by leaf node ID
        """
        self.decision_tree_object = self.tree(X_train, y_train)
        self.leaf_list, self.param_dict, self.output_dict = self.classification(self.decision_tree_object,
                                                                 X_train, y_train)
        self.LR_results = self.regressor(self.param_dict, self.output_dict)

        return #self.decision_tree_object, self.param_dict, self.output_dict, self.LR_results

    def predict(self, X_test):
        """
        Performs the prediction for each sample in the dataset (X_test). First, an initial prediction with X_test and
        the decision tree is executed to see at which leaf node each sample lands. The apply() lists contain the node
        for each sample.
        For each sample in the apply() list, the proper model is searched for in the coefficients results dictionary.
        When the correct model is found, we can gather its coefficients and intercept for the prediction as y = X*β + ε
        where
                X: the feature vector for given sample
                β: the coefficients vector for the sample's leaf node
                ε: the intercept for the sample's leaf node
                y: resulting prediction for given sample
        Each y result is attached to the linear_predictions DataFrame
        :param X_test: training/to predict dataset parameters
        :return linear_predictions: DataFrame that contains each prediction value of X_test
        """
        self.linear_predictions = pd.Series(dtype='float64', name='linear predictions')
        if hasattr(self, 'decision_tree_object'):
            #print("Using the decision tree object from fit()")
            # y_pred = self.decision_tree_object.predict(X_test) # no se si esto es necesario


            self.leaf_test_list  = self.decision_tree_object.apply(X_test)

            for leaf_node in range(len(X_test)):
                leaf_id_value = self.leaf_test_list[leaf_node]
                for model in self.LR_results:
                    #print(model['Model: '])
                    if model['Model: '] == leaf_id_value:
                        # prediction y = X * B + intercept
                        current_coefficients = model["Coefficients: "]
                        current_intercept = model['Intercept: ']
                        current_X = X_test.iloc[leaf_node]

                        y_lr_pred = np.dot(current_X, current_coefficients) + current_intercept
                        y_lr_series = pd.Series([y_lr_pred], index=[leaf_node], name='linear predictions')
                        self.linear_predictions = pd.concat([self.linear_predictions, y_lr_series])

        else:
            print("Error: fit() must be called before predict()")
        return self.linear_predictions



    def tree(self, X_train, y_train):
        """
        Builds the initial decision tree used for sample classification
        :param training_data: dataset to train the tree
        :return dtr: the decision tree object
        :X_test: test dataset
        y_test: test output variable
        """

        # df = pd.read_csv(training_data)
        # X = df.iloc[:, 0:16]
        # y = df.iloc[:, 16]
        #
        # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.5)

        self.dtr = DecisionTreeRegressor(max_depth=9, max_features=15, random_state=10)

        self.dtr.fit(X_train, y_train)

        # y_pred = dtr.predict(X_test)
        # y_pred_train = dtr.predict(X_train)

        return self.dtr

    def classification(self, dtr, X_train, y_train):
        """
        Classifies each sample depending on the leaf node of the decision tree they land in
        using the apply() method. Creates a dictionary where each node contains all its samples in a list
            leaf_params_dict = {ID1: [ [sample1],[sample2]...], ID2: [ [sample3],[sample4]...]... }
            leaf_result_dict = {ID1: [ [y1, y2]...]. ID2: [y3, y4]...}
        :param dtr: decision tree structure
        :param X_train: parameter training set
        :param y_train: prediction set
        :return leaf_params_dict: Contains all leaf nodes with its grouped sample parameters:
        :return leaf_result_dict: Contains all leaf nodes with its grouped sample predictions
        """
        self.leaf_sample_list = dtr.apply(X_train)

        self.leaf_params_dict = {}
        self.leaf_result_dict = {}

        for leaf_node in range(len(X_train)):
            leaf_id_value = self.leaf_sample_list[leaf_node]
            if leaf_id_value not in self.leaf_params_dict:
                self.leaf_params_dict[leaf_id_value] = []
            if leaf_id_value not in self.leaf_result_dict:
                self.leaf_result_dict[leaf_id_value] = []

            self.leaf_params_dict[leaf_id_value].append(X_train.iloc[leaf_node].tolist())
            self.leaf_result_dict[leaf_id_value].append(y_train.iloc[leaf_node])

        # nodo_prueba_x = []
        # for leaf_node in range(len(X_test)):
        #     if leaf_sample_list[leaf_node] == 877:
        #         nodo_prueba_x.append(X_test.iloc[leaf_node].tolist())
        #         nodo_prueba_x.append(y_train.iloc[leaf_node].tolist())
        # dfx = pd.DataFrame(nodo_prueba_x)
        # dfx.to_csv("max_error.csv", index=False)
        return self.leaf_sample_list, self.leaf_params_dict, self.leaf_result_dict

    def regressor(self, leaf_params_dict, leaf_result_dict):
        """
        Implements the linear regression for each leaf node generated in the decision tree,
        that is, every entry on the dictionary leaf_params_dict and leaf_result_dict

        :param leaf_params_dict: Dictionary with all leaf nodes and its classified samples (parameters)
        :param leaf_result_dict: Dictionary with all leaf nodes and its classified samples (outputs)
        :return LR_results: List of dictionaries with relevant information of the linear regression
            LR_results = [{"Model": ID of the leaf node,
                          "Coefficients: ": LR.coef_,
                          "Intercept: ": LR.intercept_,
                          "RMSE ML: ": difference between Label Delay and the prediction
                          "RMSE OpenLane: ": difference between Delay and Label Delay
                              }, ...]
        """
        self.LR_results = []
        counter_progress = 1
        for key, val in leaf_params_dict.items():
            # print(f"Executing n#{counter_progress} out of {len(leaf_params_dict)}")
            # print(f"Node ID: {key} \t\t Value: ", end='')
            # print(f"Depth of val: {len(val)}")
            counter_progress = counter_progress + 1

            '''
                Contador para detener la ejecución 
            '''
            # counter_progress = counter_progress + 1
            # if counter_progress > 3:
            #     break

            if (len(val) > 0):
                X_LR = leaf_params_dict[key]
                y_LR = leaf_result_dict[key]

                #X_LR_train, X_LR_test, y_LR_train, y_LR_test = train_test_split(X_LR, y_LR, test_size=0.2,
                #                                                                random_state=1)
                LR = linear_model.LinearRegression()
                #LR = Ridge(alpha=1.0)
                #OPL_delay = [sublist[3] for sublist in X_LR_test]

                LR.fit(X_LR, y_LR)
                # LR_pred = LR.predict(X_LR_test)
                # (prediccion de openlane) para ver si la del modelo puede ser menor a la de openlane


                resultado_LR = {"Model: ": key,
                                "Coefficients: ": LR.coef_,
                                "Intercept: ": LR.intercept_,
                                # "Score: ": r2_score(y_LR_test, LR_pred),
                                #"RMSE ML: ": root_mean_squared_error(y_LR_test, LR_pred),
                                #"RMSE OpenLane: ": root_mean_squared_error(OPL_delay, y_LR_test)
                                }
                self.LR_results.append(resultado_LR)
            # else:
            #
            #     self.LR_results.append(0)
            #     print("Skipped one element in linear regression")

        return self.LR_results