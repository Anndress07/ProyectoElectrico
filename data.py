import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


def data(data):
    """
    Removes unnecesary columns in the dataset
    :param data: Dataset for the model
    :return: treated.csv, dataset with only the relevant columns for the model
    """
    df = pd.read_csv(data)
    pd.set_option('display.max_columns', None)
    df10 = df
    df10 = df10.drop(df.columns[0:4], axis=1) # removes Unnamed, Path #, Descr
    df10 = df10.drop(' Network', axis=1) # removes Network
    df10 = df10.drop('Drive_cell_type', axis=1)  # removes Drive_cell_type
    df10 = df10.drop('Sink_cell_type', axis=1)  # removes Sink_cell_type
    df10 = df10.drop(df10.columns[16:159], axis=1)  # removes all logic gate type label
    df10 = df10.dropna() # Drops all rows with NaN values

    df10.to_csv('treated.csv', index=False)
    print(df10.isna().sum())

def filtering(data):
    """
    Data visualization, not significant
    :param data: treated.csv, dataset with only the relevant columns for the model
    :return:
    """
    df = pd.read_csv(data)
    pd.set_option('display.max_columns', None)
    print(df.isna().sum())
    print(df.describe())
    df_2 = df[[' Fanout',' Cap',' Slew',' Delay','Label Delay']]
    df_3 = df[['X_drive','Y_drive','X_sink','Y_sink','C_drive','C_sink','X_context','Y_context','σ(X)_context','σ(Y)_context']]
    df_4 = df[['Drive_cell_size','Sink_cell_size']]

    gran_fanout = df[df['Label Delay']>0.65]


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

    ax.set_title('Scatter Plots for Multiple X-Variables against Label Delay')
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Label Delay')
    ax.legend()
    plt.show()

def three_corners(data, corner):
    """
    Creates three different datasets depending on the corners of the transistors,
    slow, typical, fast
    :param data: treated.csv, dataset with only the relevant columns for the model
    :param corner: decides the type of filtering to apply, can be "slow", "typical, or "fast"
    :return:    One of the following: slow.csv, typical.csv, fast.csv
    """
    df = pd.read_csv(data)
    df_filtered = pd.DataFrame(columns=df.columns)
    pd.set_option('display.max_columns', None)

    if (corner == "fast"):
        df_filtered = df.loc[df.groupby(df.columns[:16].tolist())['Label Delay'].idxmin()]
        df_filtered = df_filtered.reset_index(drop=True)
        df_filtered.to_csv("fast.csv", index=False)
    elif (corner == "slow"):
        df_filtered = df.loc[df.groupby(df.columns[:16].tolist())['Label Delay'].idxmax()]
        df_filtered = df_filtered.reset_index(drop=True)
        df_filtered.to_csv("slow.csv", index=False)
    # TODO: typical filtering not working properly.
    elif (corner == "typical"):
        grouped = df.groupby(df.columns[:16].tolist())
        filtered_df = grouped.apply(get_quantile_row, quantile_value=0.5).reset_index(drop=True)
        df_filtered.to_csv("typical.csv", index=False)


def get_quantile_row(group, quantile_value=0.5):
    quantile_delay = group['Label Delay'].quantile(quantile_value)  # Get the specified quantile of 'Label Delay'
    # Get the row with the closest value to the quantile
    return group.iloc[(group['Label Delay'] - quantile_delay).abs().argsort()[:1]]


def test_three(data):
    df = pd.read_csv(data)
    pd.set_option('display.max_columns', None)

    # for row in range(len(df)):
    #     target_row = df.iloc[row,:].tolist()
    #     target_row = target_row[:-1]
    #
    #     # print(target_row)
    #     repeated = df[df.iloc[:, :16].eq(target_row).all(axis=1)]
    #     if (len(repeated) > 1):
    #         print(target_row)
    #         print(repeated)
    target_row2 = [0.0, 0.0, 0.23, 2.89, 953120.0, 696320.0, 953120.0, 696320.0, 0.004535, 0.124522,
                   946220.0, 690880.0, 0.0, 0.0, 2.0, 2.0]
    repeated = df[df.iloc[:, :16].eq(target_row2).all(axis=1)]
    print(repeated)

#data("train.csv")
#filtering('treated.csv')
#plotall('treated.csv')
three_corners('treated.csv', 'slow')
#test_three('typical.csv')

"""

"""