import pickle

#import predict as predict
from predict import readcsv

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

X_train, X_test, y_train, y_test = readcsv("slow.csv")

with open("hb_instance2.pk1", "rb") as input_file:
    hb = pickle.load(input_file)

#print(f"y_test \n {y_test}")
y_test = list(y_test)

pred_dataframe = hb.linear_predictions
pred_dataframe["y_test"] = pred_dataframe['idx on X_test'].apply(lambda x: y_test[int(x)])
pred_dataframe['error'] = abs(pred_dataframe['y_test'] - pred_dataframe['y_pred'])
pd.set_option('display.float_format', '{:.3f}'.format)
# df['y_test'] = df['idx on X_test'].apply(lambda x: y_test[int(x)])
print(pred_dataframe)

y_pred = pred_dataframe['y_pred']
#print(len(hb.linear_predictions))
#print(len(y_test))

large_error = pred_dataframe.nlargest(10, 'error')
small_error = pred_dataframe.nsmallest(10, 'error')
print(large_error)
#print(small_error)

#print(pred_dataframe.loc[24])

filtered_df = pred_dataframe[pred_dataframe['node_id'] == 598.000]

# Display the filtered DataFrame
print("\n",filtered_df)


# Plotting
import matplotlib.pyplot as plt

plt.plot(pred_dataframe['y_pred'], label='Predictions')
plt.plot(y_test, label='Actual')
#plt.ylim(-100, 100)
plt.legend()
plt.show()

print("--linear reg")
print("\tMAE test", mean_absolute_error(y_test, y_pred))
print("\tMean Squared Error (MSE) test:", mean_squared_error(y_test, y_pred))
print("\tR-squared Score test: ", r2_score(y_test, y_pred))

OPL_delay = X_test[' Delay']


# print(OPL_delay)
# print(len(y_test))
OPL_RMSE = root_mean_squared_error(OPL_delay, y_test)
ML_RMSE = root_mean_squared_error(y_pred, y_test)

print(f"\tOPL_RMSE: {OPL_RMSE}")
print(f"\tML_RMSE: {ML_RMSE}")

print(f"sample test: {X_test.iloc[9429].values}")

# print(y_test)
# print(X_test[' Delay'])

opl_rmse_list = []
ml_rmse_list = []
# for i in range(5):
#     print(f"X_test: {X_test.iloc[i][' Delay']}")
#     print(f"\ty_test: {y_test[i]}")
    #opl_rmse_list.append(root_mean_squared_error(X_test.iloc[i][" Delay"], y_test.iloc[i]))
    #ml_rmse_list.append(root_mean_squared_error(y_pred.iloc[i], y_test[i]))
# plt.plot(opl_rmse_list, label='OPL;', color='blue')
# #plt.plot(ml_rmse_list, label='ML', color='orange')
# plt.title('OPL vs ML')
# plt.xlabel('Samples')
# plt.ylabel('Values')
# plt.legend()
# plt.show()
# plt.plot(y_test.reset_index(drop=True), label='Actual', color='blue')
# plt.plot(y_pred.reset_index(drop=True), label='Predicted', color='orange')
#
# plt.title('Actual vs Predicted')
# plt.xlabel('Samples')
# plt.ylabel('Values')
# plt.legend()
# plt.show()
#
# plt.scatter(y_test, y_pred, alpha=0.5)
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Diagonal line
# plt.title('Scatter Plot of Actual vs Predicted')
# plt.xlabel('Actual')
# plt.ylabel('Predicted')
# plt.show()
#
# residuals = y_test - y_pred
# plt.scatter(y_test, residuals, alpha=0.5)
# plt.axhline(y=0, color='red', linestyle='--')
# plt.title('Residual Plot (Actual vs Residuals)')
# plt.xlabel('Actual')
# plt.ylabel('Residuals')
# plt.show()
#
# plt.hist(residuals, bins=50, alpha=0.75)
# plt.title('Histogram of Residuals')
# plt.xlabel('Residual')
# plt.ylabel('Frequency')
# plt.show()
#
# import scipy.stats as stats
# import matplotlib.pyplot as plt
#
# stats.probplot(residuals, dist="norm", plot=plt)
# plt.title('QQ Plot of Residuals')
# plt.show()

# plt.scatter(feature, residuals, alpha=0.5)
# plt.axhline(y=0, color='red', linestyle='--')
# plt.title('Residuals by Feature')
# plt.xlabel('Feature')
# plt.ylabel('Residuals')
# plt.show()