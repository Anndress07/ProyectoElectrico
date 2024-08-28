import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import precision_score


def main():
    df = pd.read_csv("treated.csv")
    pd.set_option('display.max_columns', None)
    #print(df.head(10))
    X = df.iloc[:, 0:16]
    #print(X)
    y = df.iloc[:, 16]

    #print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.5)
    #print("X TRAIN: ", X_train)
    #print("X TEST:", X_test)



    dtr = DecisionTreeRegressor(max_depth=9, max_features=15, max_leaf_nodes=60)
    #print(dtr.get_params())
    dtr.fit(X_train, y_train)

    y_pred = dtr.predict(X_test)
    y_pred_train = dtr.predict(X_train)
    print("MAE test", mean_absolute_error(y_test, y_pred))
    print("MAE training", mean_absolute_error(y_train, y_pred_train))

    print("Mean Squared Error (MSE) test:", r2_score(y_test, y_pred))
    print("R-squared Score test: ", mean_squared_error(y_test, y_pred))
    #print("accuracy score: ", precision_score(y_test, y_pred))

    print("y_pred_train", y_pred_train)
    """ 
    from sklearn.model_selection import GridSearchCV
    parameters = {'max_depth': [6,  9,  12], 'max_leaf_nodes': [36,  48,  52],
                  'max_features': [10, 14, 18]}
    # {'max_depth': 9, 'max_features': 18, 'max_leaf_nodes': 52}
    rg1 = DecisionTreeRegressor()
    rg1 = GridSearchCV(rg1, parameters)
    rg1.fit(X_train, y_train)
    print(rg1.best_params_)
    """

    #print(dtr.feature_importances_)
    features = pd.DataFrame(dtr.feature_importances_, index=X.columns)
    features.head(16).plot(kind='bar')

    tree.plot_tree(dtr)
    plt.show()
    return dtr, X_test



def treeStructure(dtr, X_test):
    n_nodes = dtr.tree_.node_count
    children_left = dtr.tree_.children_left
    children_right = dtr.tree_.children_right
    feature = dtr.tree_.feature
    threshold = dtr.tree_.threshold
    values = dtr.tree_.value

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
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

    print(
        "The binary tree structure has {n} nodes and has "
        "the following tree structure:\n".format(n=n_nodes)
    )
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

    print("Rules used to predict sample {id}:\n".format(id=sample_id))
    for node_id in node_index:
        # continue to the next node if it is a leaf node
        if leaf_id[sample_id] == node_id:
            continue

        # check if value of the split feature for sample 0 is below threshold
        if X_test[sample_id, feature[node_id]] <= threshold[node_id]:
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        print(
            "decision node {node} : (X_test[{sample}, {feature}] = {value}) "
            "{inequality} {threshold})".format(
                node=node_id,
                sample=sample_id,
                feature=feature[node_id],
                value=X_test[sample_id, feature[node_id]],
                inequality=threshold_sign,
                threshold=threshold[node_id],
            )
        )


if __name__ == "__main__":
    dtr, X_test = main()
    treeStructure(dtr, X_test)

