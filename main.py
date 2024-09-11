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


def main():
    df = pd.read_csv("treated.csv")
    pd.set_option('display.max_columns', None)
    #print(df.head(10))
    X = df.iloc[:, 0:16]
    #print(X)
    y = df.iloc[:, 16]

    #print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.5)
    #X_test.to_csv('X_test.csv', index=False)
    #y_test.to_csv('y_test.csv', index=False)
    #print("X TRAIN: ", X_train)
    #print("X TEST:", X_test)



    dtr = DecisionTreeRegressor(max_depth=9, max_features=15, random_state=10)
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
    #features = pd.DataFrame(dtr.feature_importances_, index=X.columns)
    #features.head(16).plot(kind='bar')
    #plt.show()

    #tree.plot_tree(dtr)
    #plt.show()

    return dtr, X_test, y_test



def treeStructure(dtr, X_test, enable):
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


def classification(dtr, total_leaves, X_test,y_test):
    leaf_sample_list = dtr.apply(X_test)
    print(f"leaf_sample_list len: {len(leaf_sample_list)}")
    print(f"leaf_sample_list: {leaf_sample_list}")

    leaf_params_dict = {}
    leaf_result_dict = {}
    nodo_prueba_x = []
    nodo_prueba_y = []

    for leaf_node in range(len(X_test)):
        # if leaf_sample_list[leaf_node] == 767:
        #     nodo_prueba_x.append(X_test.iloc[leaf_node].tolist())
        #     nodo_prueba_y.append(y_test.iloc[leaf_node].tolist())

        leaf_id_value = leaf_sample_list[leaf_node]
        if leaf_id_value not in leaf_params_dict:
            leaf_params_dict[leaf_id_value] = []
        if leaf_id_value not in leaf_result_dict:
            leaf_result_dict[leaf_id_value] = []


        #leaf_params_dict[leaf_id_value].append(X_test[leaf_node])
        leaf_params_dict[leaf_id_value].append(X_test.iloc[leaf_node].tolist())
        leaf_result_dict[leaf_id_value].append(y_test.iloc[leaf_node])

    #nodo_prueba_x = pd.DataFrame(nodo_prueba_x)

    #nodo_prueba_x['16'] = nodo_prueba_y
    #print(f"nodo_prueba_x: {nodo_prueba_x}")
    #print(leaf_params_dict)
    #nodo_prueba_x.to_csv('prueba_lnr.csv', index=False)

    #leaf_params_dict[767].append(X_test.iloc[2].tolist())
    #leaf_result_dict[767].append(y_test.iloc[2])
    #print(type(leaf_params_dict[767]))
    #print(leaf_params_dict[767])
    return leaf_sample_list, leaf_params_dict,leaf_result_dict

def regressor(leaf_sample_list, total_leaves, leaf_params_dict, leaf_result_dict, data):
    df = pd.read_csv("treated.csv")

    LR_results = []
    counter_progress = 1
    for key,val in leaf_params_dict.items():
        print(f"Executing n#{counter_progress} out of {len(leaf_params_dict)}")
        print(f"Node ID: {key} \t\t Value: ", end='')
        print(f"Depth of val: {len(val)}")
        idx_counter = 0
        ''' Secuencia para verificar que los diccionarios con los parámetros leaf_params_dict y resultados 
            leaf_result_dict estén alineados con respecto a los samples originales. 
        '''
        # for val_in_node in val:
        #     target_row = val_in_node
        #     target_row.append(leaf_result_dict[key][idx_counter])
        #     filtered_df = df[df.iloc[:, :17].eq(target_row).all(axis=1)]
        #     idx_counter = idx_counter + 1
        #     if  filtered_df.empty:
        #         print(f"FAIL")
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
            resultado_LR = {"Model: ": key,
                          #"Coefficients: ": LR.coef_,
                          #"Intercept: ": LR.intercept_,
                          #"Score: ": r2_score(y_LR_test, LR_pred),
                          "RMSE ML: ": root_mean_squared_error(y_LR_test, LR_pred),
                          "RMSE OpenLane: ": root_mean_squared_error(OPL_delay, y_LR_test)
                          }
            LR_results.append(resultado_LR)

    score_results = []
    mse_results = []
    ml_hist = []
    opl_hist = []
    for item in LR_results:
        ml_hist.append(item['RMSE ML: '])
        print(f"error {item['RMSE ML: ']}")
        # if (item['RMSE ML: '] > 10):
        #print(item['RMSE ML: '])
        if (item['RMSE ML: ']) < 10:
            score_results.append(item['RMSE ML: '])
        mse_results.append(item['RMSE OpenLane: '])
        #mse_results.append(item['RMSE OpenLane: '])
    print(f"The ML error is: {sum(score_results) / len(score_results)}")
    print(f"The OPL error is: {sum(mse_results) / len(mse_results)}")
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
    plt.show()

    fig2, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    # Step 3: Plot the histograms in subplots
    # Histogram for 'RMSE ML'
    axs[0].hist(ml_hist, bins=10, color='red', edgecolor='black')
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
    plt.show()


    return


if __name__ == "__main__":
    dtr, X_test,y_test = main()
    total_leaves = treeStructure(dtr, X_test, 0)
    print("total_leaves: ", total_leaves)
    leaf_sample_list, leaf_params_dict, leaf_result_dict = classification(dtr, total_leaves, X_test,y_test)
    regressor(leaf_sample_list, total_leaves, leaf_params_dict, leaf_result_dict, "treated.csv")

