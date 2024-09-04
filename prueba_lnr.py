import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import precision_score


df = pd.read_csv("prueba_lnr.csv")
pd.set_option('display.max_columns', None)
#print(df.head(10))
X = df.iloc[:, 0:16]
#print(X)
y = df.iloc[:, 16]

X_LR_train, X_LR_test, y_LR_train, y_LR_test = train_test_split(X, y, test_size=0.2)

LR = linear_model.LinearRegression()

LR.fit(X_LR_train, y_LR_train)
LR_pred = LR.predict(X_LR_test)
print(f"score:  {LR.score(X_LR_test, y_LR_test)}")


print(f"r2: {r2_score(y_LR_test, LR_pred)}")
print(f"MAE: : {mean_squared_error(y_LR_test, LR_pred)}")

plt.scatter(X_LR_test.iloc[:, 3], y_LR_test, color="black")
plt.plot(
    X_LR_test.iloc[:, 3], LR_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()