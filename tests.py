from hybridmodel import HybridModel


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
y_pred, y_lr_pred = hb.predict(X_test, y_test)
print(y_test)
print(y_lr_pred)


print("--tree")
print("MAE test", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE) test:", r2_score(y_test, y_pred))
print("R-squared Score test: ", mean_squared_error(y_test, y_pred))

print("--linear reg")
print("MAE test", mean_absolute_error(y_test, y_lr_pred))
print("Mean Squared Error (MSE) test:", r2_score(y_test, y_lr_pred))
print("R-squared Score test: ", mean_squared_error(y_test, y_lr_pred))





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

treeStructure(decision_tree_object, X_test, 0)
print(param_dict[877])
