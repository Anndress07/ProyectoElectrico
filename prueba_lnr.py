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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import Ridge
#from predict import readcsv
scaler = StandardScaler()
df = pd.read_csv("prueba_lnr.csv")
pd.set_option('display.max_columns', None)

def readcsv(training_data):
    df = pd.read_csv(training_data)
    X = df.iloc[:, 0:16]
    y = df.iloc[:, 16]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.5)
    #print(f"y_Test {y_test}")
    #print(f"x_Test: {X_test}")
    y_test = y_test.head(10000)
    X_test = X_test.head(10000)
    pd.set_option('display.max_columns', None)
    # print(f"y_Test {y_test}")
    # print(f"x_Test: {X_test}")
    return X_train, X_test, y_train, y_test
#print(df.describe().round(3))

X_train, X_test, y_train, y_test = readcsv("slow.csv")

# STD
X1 = df.iloc[:,0:17]
X1 = scaler.fit_transform(X1)
X1 = pd.DataFrame(X1, columns=df.columns)
#print(X1.describe().round(3))

# Normalization
scaleMinMax = MinMaxScaler(feature_range=(0,1))
X2 = df.iloc[:,0:17]
X2 = scaleMinMax.fit_transform(X2)
X2 = pd.DataFrame(X2, columns=df.columns)
#print(X2.describe().round(3))



#print(df.head(10))
X = df.iloc[:, 0:16]
#print(X)
y = df.iloc[:, 16]
X_LR_train = X1
y_LR_train = y
X_train_scaled = scaler.fit_transform(X_train)
# 0.00000000e+00 0.00000000e+00 1.20000000e-01 1.17000000e+00
#  3.31660000e+05 3.59040000e+05 3.31660000e+05 3.59040000e+05
#  4.56400000e-03 1.34592000e-01 3.19700000e+05 4.78720000e+05
#  1.59348674e+03 1.06010233e+05 2.00000000e+00 2.00000000e+00
X_LR_test = X_test.iloc[9429].values



y_LR_test = y_test.iloc[9429]

#X_LR_train, X_LR_test, y_LR_train, y_LR_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(f"X_lr_test {X_LR_test}")
print(f"y_lr_test {y_LR_test}")
# X_LR_train = scaler.fit_transform(X_LR_train)
# X_LR_test = scaler.transform(X_LR_test)
#

#
# LR.fit(X_LR_train, y_LR_train)
# LR_pred_train = LR.predict(X_LR_train)
# LR_pred = LR.predict(X_LR_test)

# LR = linear_model.LinearRegression()
LR = Ridge(alpha=1.0)
X_LR_train = scaler.fit_transform(X_LR_train)
X_LR_test = scaler.fit_transform(X_LR_test.reshape(1, -1))
LR.fit(X_LR_train, y_LR_train)
LR_pred_train = LR.predict(X_LR_train)
LR_pred = LR.predict(X_LR_test.reshape(1, -1))

# LR = Ridge(alpha=1.0)
# LR.fit(X, y)

print(f"y_pred: {LR_pred}")

# print(f"score:  {LR.score(X_LR_test, y_LR_test)}")
#
# print("test--")
# print(f"\tr2: {r2_score(y_LR_test, LR_pred)}")
# print(f"\tMAE: : {mean_squared_error(y_LR_test, LR_pred)}")
# print(f"\tRMSE ML: { root_mean_squared_error(y_LR_test, LR_pred)}")
# print(f"\tpred: {LR_pred}")
#
# #print(f"\tdata: {X_LR_test.describe().round(3)}")
# print("train--")
# print(f"\tr2: {r2_score(y_LR_train, LR_pred_train)}")
# print(f"\tMAE: : {mean_squared_error(y_LR_train, LR_pred_train)}")
# print(f"\tRMSE ML: { root_mean_squared_error(y_LR_train, LR_pred_train)}")
# print(f"\tpred: {LR_pred_train}")

# print(X_LR_train.iloc[1, 0:16])
# print(X_LR_test.iloc[0, 0:16])
#print(f"\tcoef: {LR.coef_}")
#print(f"\tdata: {X_LR_train.describe().round(3)}")
# print(f"coefs:")
# print(LR.coef_)
# print(type(LR.coef_))

for i in range(len(LR.coef_)):
    print(f"\tParameter: {df.columns[i]}, coef: {LR.coef_[i]}")
    #print(LR.coef_[i])

# plt.scatter(X_LR_test.iloc[:, 3], y_LR_test, color="black")
# plt.plot(
#     X_LR_test.iloc[:, 3], LR_pred, color="blue", linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()
# print(f"lr pred {LR_pred}")
#plt.scatter(LR_pred, y_LR_test, color="blue", label="test")
plt.scatter(LR_pred_train, y_LR_train, color="purple", label="train" )
#plt.plot([y_LR_test.min(), y_LR_test.max()], [y_LR_test.min(), y_LR_test.max()], color="red", linewidth=2)
plt.plot([y_LR_train.min(), y_LR_train.max()], [y_LR_train.min(), y_LR_train.max()], color="green" , linewidth=2)
plt.legend()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()