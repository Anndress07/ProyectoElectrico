import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


def data(data):
    df = pd.read_csv(data)
    pd.set_option('display.max_columns', None)
    df10 = df
    df10 = df10.drop(df.columns[0:4], axis=1) # borra Unnamed, Path #, Descr
    df10 = df10.drop(' Network', axis=1) # Borra Network
    df10 = df10.drop('Drive_cell_type', axis=1)  # Borra Drive_cell_type
    df10 = df10.drop('Sink_cell_type', axis=1)  # Borra Sink_cell_type
    df10 = df10.drop(df10.columns[16:159], axis=1)  # Borra todos los tipos de compuerta
    #print(df10)
    df10.to_csv('treated.csv', index=False)
    print(df10.isna().sum())

def filtering(data):
    df = pd.read_csv(data)
    pd.set_option('display.max_columns', None)
    #print(df.isna().sum())
    #print(df[df[' Fanout'].isna()])
    #print(df.describe())
    #df.hist(bins=60)
    #plt.show()

    #print(df[df[' Fanout']>1])
    gran_fanout = df[df[' Fanout']>0]
    gran_fanout[[' Fanout', 'Label Delay']].plot(kind="scatter",x=' Fanout', y='Label Delay', figsize=(20,10))
    plt.show()

#data("train.csv")
filtering('treated.csv')
