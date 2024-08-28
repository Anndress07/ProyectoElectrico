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
    #df10 = df10.dropna(subset=['Label Delay'])
    df10 = df10.dropna() # BOTAR TODAS LAS FILAS CON UN VALOR NaN
    #print(df10)
    df10.to_csv('treated.csv', index=False)
    print(df10.isna().sum())

def filtering(data):
    df = pd.read_csv(data)
    pd.set_option('display.max_columns', None)
    #print("NaN en el dataset:")
    print(df.isna().sum())
    #print(df[df[' Fanout'].isna()])
    print(df.describe())

    df_2 = df[[' Fanout',' Cap',' Slew',' Delay','Label Delay']]
    df_3 = df[['X_drive','Y_drive','X_sink','Y_sink','C_drive','C_sink','X_context','Y_context','σ(X)_context','σ(Y)_context']]
    df_4 = df[['Drive_cell_size','Sink_cell_size']]
    #print(df_2)
    #df_4.hist(bins=60)
    #plt.show()

    #print(df[df[' Fanout']>1])
    gran_fanout = df[df['Label Delay']>0.65]
    #gran_fanout[[' Fanout',' Cap', 'Label Delay']].plot(kind="scatter",x=' Fanout', y='Label Delay', figsize=(5,5))
    #df.plot(kind="scatter",x="Fanout",y='Label Delay',color='blue',label='Label Delay')
    #plt.title('Height vs. Weight and Age')
    #plt.xlabel('Height')
    #plt.ylabel('Value')

    fig, ax = plt.subplots(figsize=(20, 20))
    """ 
    # First scatter plot: 'Fanout' vs. 'Label Delay'
    gran_fanout.plot(kind='scatter', x=' Fanout', y='Label Delay', color='blue', label='Fanout', ax=ax)
    # Second scatter plot: 'Another Fanout' vs. 'Label Delay'
    gran_fanout.plot(kind='scatter', x=' Cap', y='Label Delay', color='red', label='Cap', ax=ax)
    gran_fanout.plot(kind='scatter', x=' Slew', y='Label Delay', color='black', label='Slew', ax=ax)
    gran_fanout.plot(kind='scatter', x=' Delay', y='Label Delay', color='yellow', label='Delay', ax=ax)
    """
    # First scatter plot: 'Fanout' vs. 'Label Delay'
    gran_fanout.plot(kind='scatter', x='X_drive', y='Label Delay', color='blue', label='X_drive', ax=ax)
    # Second scatter plot: 'Another Fanout' vs. 'Label Delay'
    gran_fanout.plot(kind='scatter', x='Y_drive', y='Label Delay', color='red', label='Y_drive', ax=ax)
    gran_fanout.plot(kind='scatter', x='X_sink', y='Label Delay', color='black', label='X_sink', ax=ax)
    gran_fanout.plot(kind='scatter', x='Y_sink', y='Label Delay', color='yellow', label='Y_sink', ax=ax)

    # Customize the plot
    ax.set_title('Scatter Plots for Multiple X-Variables against Label Delay')
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Label Delay')
    ax.legend()

    # Show the plot
    plt.show()

def plotall(data):
    df = pd.read_csv(data)
    df.set_index('Label Delay', inplace=True)

    # Plot the entire DataFrame with lines
    df = df.iloc[::50000, :]
    df.plot(kind='line', figsize=(10, 6))

    # Customize the plot
    plt.title('Line Plot for Entire DataFrame')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend(title='Variables')
    plt.grid(True)

    # Show the plot
    plt.show()

#data("train.csv")
filtering('treated.csv')
#plotall('treated.csv')
#hello

"""

"""