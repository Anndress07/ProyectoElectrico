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


def tree(data):
    """
    Builds the initial decision tree used for sample classification
    :param data: dataset to train the tree
    :return dtr: the decision tree object
    :X_test: test dataset
    y_test: test output variable
    """
    df = pd.read_csv(data)
    pd.set_option('display.max_columns', None)
    X = df.iloc[:, 0:16]
    y = df.iloc[:, 16]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.5)

    dtr = DecisionTreeRegressor(max_depth=9, max_features=15, random_state=10)

    dtr.fit(X_train, y_train)

    y_pred = dtr.predict(X_test)
    y_pred_train = dtr.predict(X_train)

    print("MAE test", mean_absolute_error(y_test, y_pred))
    print("MAE training", mean_absolute_error(y_train, y_pred_train))
    print("Mean Squared Error (MSE) test:", r2_score(y_test, y_pred))
    print("R-squared Score test: ", mean_squared_error(y_test, y_pred))
    # print("accuracy score: ", precision_score(y_test, y_pred))

    """
    GridSearch optimal parameter max_features': [10, 14, 18]
    TODO: run more parameters with more cpu
     """
    # from sklearn.model_selection import GridSearchCV
    # parameters = {'max_depth': [6,  9,  12], 'max_leaf_nodes': [36,  48,  52],
    #               'max_features': [10, 14, 18]}
    # # {'max_depth': 9, 'max_features': 18, 'max_leaf_nodes': 52}
    # rg1 = DecisionTreeRegressor()
    # rg1 = GridSearchCV(rg1, parameters)
    # rg1.fit(X_train, y_train)
    # print(rg1.best_params_)

    # Feature importances graph
    print(dtr.feature_importances_)
    features = pd.DataFrame(dtr.feature_importances_, index=X.columns)
    features.head(16).plot(kind='bar')
    # plt.show() # TODO: undo comment

    # Diagram of the tree
    # tree.plot_tree(dtr)
    # plt.show()

    return dtr, X_test, y_test


def main():
    dtr, X_test, y_test = tree('slow.csv')
    total_leaves = treeStructure(dtr, X_test, 0)
    # print("total_leaves: ", total_leaves)
    leaf_sample_list, leaf_params_dict, leaf_result_dict = classification(dtr, X_test, y_test)
    lr_results = regressor(leaf_params_dict, leaf_result_dict)
    regressor_results(lr_results, leaf_params_dict, leaf_result_dict)


def treeStructure(dtr, X_test, enable):
    """
    Iterates through the entire tree structure, can display its contents depending on the value
    of "enable".
    :param dtr: decision tree structure
    :param X_test: parameter training set
    :param enable: controls the level of detailedness in the displaying of information
            enable == 1: prints the tree structure (whether if its a split or leaf node, along with its ID)
            enable == 2: prints the path used to predict a specific sample
            enable == 3: display both tree structure and prediction path
    :return total_leaf_nodes: total amount of leaf nodes in the tree
    """
    n_nodes = dtr.tree_.node_count
    children_left = dtr.tree_.children_left
    children_right = dtr.tree_.children_right
    feature = dtr.tree_.feature
    threshold = dtr.tree_.threshold
    values = dtr.tree_.value

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    total_leaf_nodes = 0
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
            total_leaf_nodes = total_leaf_nodes + 1

    print(
        "The binary tree structure has {n} nodes and has "
        "the following tree structure:\n".format(n=n_nodes)
    )
    if (enable == 1 or enable == 3):
        for i in range(n_nodes):
            if is_leaves[i]:
                print(
                    "{space}node={node} is a leaf node with value={value}.".format(
                        space=node_depth[i] * "\t", node=i, value=values[i]
                    )
                )
            else:
                print(
                    "{space}node={node} is a split node with value={value}: "
                    "go to node {left} if X[:, {feature}] <= {threshold} "
                    "else to node {right}.".format(
                        space=node_depth[i] * "\t",
                        node=i,
                        left=children_left[i],
                        feature=feature[i],
                        threshold=threshold[i],
                        right=children_right[i],
                        value=values[i],
                    )
                )
    node_indicator = dtr.decision_path(X_test)
    leaf_id = dtr.apply(X_test)

    sample_id = 0
    # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
    node_index = node_indicator.indices[
                 node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]
                 ]
    if (enable == 2 or enable == 3):
        print("Rules used to predict sample {id}:\n".format(id=sample_id))
        for node_id in node_index:
            # continue to the next node if it is a leaf node
            if leaf_id[sample_id] == node_id:
                continue

            # check if value of the split feature for sample 0 is below threshold
            if X_test.iloc[sample_id, feature[node_id]] <= threshold[node_id]:
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            print(
                "decision node {node} : (X_test[{sample}, {feature}] = {value}) "
                "{inequality} {threshold})".format(
                    node=node_id,
                    sample=sample_id,
                    feature=feature[node_id],
                    value=X_test.iloc[sample_id, feature[node_id]],
                    inequality=threshold_sign,
                    threshold=threshold[node_id],
                )
            )

    return total_leaf_nodes


def classification(dtr, X_test, y_test):
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


def regressor(leaf_params_dict, leaf_result_dict):
    """
    Implements the linear regression for each leaf node generated in the decision tree,
    that is, every entry on the dictionary leaf_params_dict and leaf_result_dict

    :param leaf_params_dict: Dictionary with all leaf nodes and its classified samples (parameters)
    :param leaf_result_dict: Dictionary with all leaf nodes and its classified samples (outputs)
    :return LR_results: Dictionary with relevant information of the linear regression
        LR_results = {"Model": ID of the leaf node,
                      "Coefficients: ": LR.coef_,
                      "Intercept: ": LR.intercept_,
                      "RMSE ML: ": difference between Label Delay and the prediction
                      "RMSE OpenLane: ": difference between Delay and Label Delay
                          }
    """
    LR_results = []
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

        if (len(val) > 1):
            X_LR = leaf_params_dict[key]
            y_LR = leaf_result_dict[key]

            X_LR_train, X_LR_test, y_LR_train, y_LR_test = train_test_split(X_LR, y_LR, test_size=0.2, random_state=1)
            LR = linear_model.LinearRegression()
            OPL_delay = [sublist[3] for sublist in X_LR_test]

            LR.fit(X_LR_train, y_LR_train)
            LR_pred = LR.predict(X_LR_test)
            # (prediccion de openlane) para ver si la del modelo puede ser menor a la de openlane
            if key == 877:
                print(f"y_LR_test: {y_LR_test}\n"
                      f"X_LR_test: {X_LR_test}\n"
                      f"LR_pred: {LR_pred}")
            resultado_LR = {"Model: ": key,
                            # "Coefficients: ": LR.coef_,
                            # "Intercept: ": LR.intercept_,
                            # "Score: ": r2_score(y_LR_test, LR_pred),
                            "RMSE ML: ": root_mean_squared_error(y_LR_test, LR_pred),
                            "RMSE OpenLane: ": root_mean_squared_error(OPL_delay, y_LR_test)
                            }
            LR_results.append(resultado_LR)

    return LR_results


def regressor_results(LR_results, leaf_params_dict, leaf_result_dict):
    score_results = []
    mse_results = []
    ml_hist = []

    for item in LR_results:
        ml_hist.append(item['RMSE ML: '])
        #print(f"error {item['RMSE ML: ']}")
        # if (item['RMSE ML: '] > 10):
        #print(item['RMSE ML: '])
        #if (item['RMSE ML: ']) < 100:
        score_results.append(item['RMSE ML: '])
        mse_results.append(item['RMSE OpenLane: '])
        #mse_results.append(item['RMSE OpenLane: '])
    print(f"The ML error is: {sum(score_results) / len(score_results)}")
    print(f"The OPL error is: {sum(mse_results) / len(mse_results)}")

    max_rmse_ml = max(LR_results, key=lambda x: x["RMSE ML: "])
    print(f"max rmse: {max_rmse_ml}")
    # max rmse: {'Model: ': 877, 'RMSE ML: ': 455.5338674560883, 'RMSE OpenLane: ': 1.4957523190689026}

    # for key, value in leaf_params_dict.items():
    #     print(f"Key {key}:", end='')
    #     for sublist in value[:5]:  # Slicing to get the first 5 lists
    #         print(sublist)
    #     print()
    df1 = pd.read_csv("slow.csv")
    # print(df1.columns)
    df = pd.DataFrame()
    # print(leaf_result_dict[877])
    if 877 in leaf_params_dict:
        data_to_append = []
        cnt = 0
        for sublist in leaf_params_dict[877]:
            sublist.append(leaf_result_dict[877][cnt])
            data_to_append.append(sublist)
            cnt += 1
            #max_error_df = max_error_df._append(sublist, ignore_index=True)
            # print(sublist)

        df = pd.concat([df, pd.DataFrame(data_to_append)], ignore_index=True)
    #     print(df)
    # print(LR_results)



    df.to_csv("max_error.csv", index=False)

    #plt.plot(score_results, color="red", label="RMSE ML")
    #plt.plot(mse_results, color="blue", label="RMSE OpenLane")
    #plt.show()
    #print(LR_results)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

    # Plot the first graph on the first subplot
    ax1.plot(score_results, color="red", label="RMSE ML 1")
    ax1.set_title("Plot 1")
    ax1.legend()

    # Plot the second graph on the second subplot
    ax2.plot(mse_results, color="blue", label="RMSE OpenLane: ")
    ax2.set_title("Plot 2")
    ax2.legend()

    ax3.plot(score_results, color="red", label="RMSE ML 1")
    ax3.plot(mse_results, color="blue", label="RMSE OpenLane: ")
    ax3.set_title("Plot 3")
    ax3.legend()
    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plots
    # plt.show() # TODO: undo comment

    fig2, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    # Step 3: Plot the histograms in subplots
    # Histogram for 'RMSE ML'
    axs[0].hist(score_results, bins=100, color='red', edgecolor='black')
    #axs[0].set_xlim([0, 25])
    axs[0].set_title('RMSE ML Histogram')
    axs[0].set_xlabel('RMSE Value')
    axs[0].set_ylabel('Frequency')

    # Histogram for 'RMSE OpenLane'
    axs[1].hist(mse_results, bins=100, color='blue', edgecolor='black')
    axs[1].set_title('RMSE OpenLane Histogram')
    axs[1].set_xlabel('RMSE Value')
    axs[1].set_ylabel('Frequency')

    # Step 4: Adjust layout and show the plot
    plt.tight_layout()
    # plt.show() # TODO: undo comment
    return


if __name__ == "__main__":
    main()
