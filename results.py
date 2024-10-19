import pickle

#import predict as predict
from predict import readcsv_p

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

X_train, X_test, y_train, y_test = readcsv_p("slow.csv", 0)



with open("hb_instance2.pk1", "rb") as input_file:
    hb = pickle.load(input_file)

#print(f"y_test \n {y_test}")
y_test = list(y_test)

pred_dataframe = hb.linear_predictions
pred_dataframe["y_test"] = pred_dataframe['idx on X_test'].apply(lambda x: y_test[int(x)])
pred_dataframe['error'] = abs(pred_dataframe['y_test'] - pred_dataframe['y_pred'])
pd.set_option('display.float_format', '{:.3f}'.format)
# df['y_test'] = df['idx on X_test'].apply(lambda x: y_test[int(x)])


# print(pred_dataframe)


y_pred = pred_dataframe['y_pred']
#print(len(hb.linear_predictions))
#print(len(y_test))
#pd.set_option('display.max_rows', None)
large_error = pred_dataframe.nlargest(4000, 'error')
small_error = pred_dataframe.nsmallest(50000, 'error')
print(f"largest errors:\n {large_error}")
print(f"smallest errors:\n {small_error}")
#print(small_error)

#print(pred_dataframe.loc[24])

filtered_df = pred_dataframe[pred_dataframe['node_id'] == 660.000]

# Display the filtered DataFrame
print("\n",filtered_df)

#  sample_to_test = [23960, 25870, 56097, 42310, 15001]
rows_to_display = pred_dataframe.iloc[[23960, 25870, 56097, 42310, 15001]]
# print(rows_to_display)



# Plotting


indices_to_remove = pred_dataframe[pred_dataframe['error'] > 10]['idx on X_test'].tolist()
for idx in indices_to_remove:
    if idx in y_pred:
        y_pred.drop(idx)


rows_before = len(pred_dataframe)
pred_dataframe.drop(pred_dataframe[pred_dataframe['error'] > 10].index, inplace=True)
rows_after = len(pred_dataframe)
rows_removed = rows_before - rows_after
print(f"Number of instances removed: {rows_removed}")


plt.plot(pred_dataframe['y_pred'], label='Predictions')
plt.plot(pred_dataframe['y_test'], label='Actual')
#plt.ylim(-100, 100)
plt.legend()
plt.show()

print("--linear reg")
print("\tMAE test", mean_absolute_error(pred_dataframe['y_test'], pred_dataframe['y_pred']))
print("\tMean Squared Error (MSE) test:", mean_squared_error(pred_dataframe['y_test'], pred_dataframe['y_pred']))
print("\tR-squared Score test: ", r2_score(pred_dataframe['y_test'], pred_dataframe['y_pred']))

OPL_delay = X_test[' Delay']


# print(OPL_delay)
# print(len(y_test))
OPL_RMSE = root_mean_squared_error(OPL_delay, y_test)
ML_RMSE = root_mean_squared_error(pred_dataframe['y_test'], pred_dataframe['y_pred'])

print(f"\tOPL_RMSE: {OPL_RMSE}")
print(f"\tML_RMSE: {ML_RMSE}")

if __name__ == "__main__":
    #X_train, X_test, y_train, y_test = readcsv_p(test_data, 0)


    #y_lr_pred = hb.predict(X_test)

    with open("hb_instance2.pk1", "wb") as output_file:
        pickle.dump(hb, output_file)
    print("Predict executed directly")


# print(f"sample test: {X_test.iloc[23960].values}")

