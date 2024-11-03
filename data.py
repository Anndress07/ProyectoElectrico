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
    col_names = ['Design', ' Fanout', ' Cap', ' Slew', ' Delay', 'X_drive', 'Y_drive', 'X_sink',
                 'Y_sink', 'C_drive', 'C_sink', 'X_context', 'Y_context', 'σ(X)_context',
                 'σ(Y)_context', 'Drive_cell_size', 'Sink_cell_size', 'Label Delay']
    df = df[col_names]
    df = df.dropna()
    df.to_csv('treated_labels.csv', index=False)
    # print(df)



    # print(df.isna().sum())

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
        print(df_filtered)
        df_filtered.to_csv("labels_slow.csv", index=False)
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

def remove_context_features(data_path):
    """
    Removes the features X_context and Y_context in both the training and testing data.
    Alters the file path name with the modified csv.
    :param train_data: file name of the training data
    :param test_data: file name of the testing data
    :return: New path names including the modified csv
    """
    df_data = pd.read_csv(data_path)
    df_data = df_data.drop(columns=['X_context', 'Y_context'], errors='ignore')
    df_data.to_csv("modded_data.csv", index=False)
    NEW_DATA = "modded_data.csv"

    return NEW_DATA

def remove_std_dvt_context(data_path):
    """
    Removes the features σ(X)_context and σ(Y)_context in both the training and testing data.
    Alters the file path name with the modified csv.
    :param train_data: file name of the training data
    :param test_data: file name of the testing data
    :return: New path names including the modified csv
    """

    df_data = pd.read_csv(data_path)
    df_data = df_data.drop(columns=['σ(X)_context', 'σ(Y)_context'], errors='ignore')
    df_data.to_csv("modded_data.csv", index=False)
    NEW_DATA = "modded_data.csv"

    return NEW_DATA


def calc_distance_parameter(data_path):
    """
    Calculates the Euclidean distance given X_drive, X_sink, Y_drive, Y_sink. Removes
    the aforementioned, adds a new parameter called "Distance"
    :param train_data: file name of the training data
    :param test_data: file name of the testing data
    :return: New path names including the modified csv
    """

    df_data = pd.read_csv(data_path)
    df_data['Distance'] = np.sqrt((df_data['X_drive'] - df_data['X_sink']) ** 2 + (df_data['Y_drive'] - df_data['Y_sink']) ** 2)
    df_data = df_data.drop(columns=['X_drive', 'Y_drive', 'X_sink', 'Y_sink'])
    cols = df_data.columns.tolist()
    cols.remove('Distance')
    delay_idx = cols.index(' Delay')
    new_cols_order = cols[:delay_idx + 1] + ['Distance'] + cols[delay_idx + 1:]
    df_data = df_data[new_cols_order]

    df_data.to_csv("modded_data.csv", index=False)
    NEW_DATA = "modded_data.csv"

    return NEW_DATA




"""

"""

if __name__ == "__main__":
    data("test_labels.csv")
    # filtering('treated.csv')
    # plotall('treated.csv')
    three_corners('treated_labels.csv', 'slow')
    # test_three('typical.csv')
