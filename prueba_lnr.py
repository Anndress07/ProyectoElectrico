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

df = pd.read_csv("prueba_lnr.csv")
pd.set_option('display.max_columns', None)

training_data = "slow.csv"

scaler = StandardScaler()
df = pd.read_csv("slow.csv")
pd.set_option('display.max_columns', None)

# STD
X1 = df.iloc[:,0:17]
X1 = scaler.fit_transform(X1)
X1 = pd.DataFrame(X1, columns=df.columns)
#print(X1.describe().round(3))