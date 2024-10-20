import random
import pickle
import numpy as np

import pandas as pd

from hybridmodel import HybridModel
from train import readcsv
from predict import readcsv_p
from results_run import build_df_imported



NUMBER_OF_RUNS = 5
TRAINING_DATA = "slow.csv"
TESTING_DATA = "slow.csv"
C_TRAINING_DATA = TRAINING_DATA
C_TESTING_DATA = TESTING_DATA
modded_train = pd.read_csv(TRAINING_DATA)
modded_test = pd.read_csv(TESTING_DATA)
modded_train.to_csv("modded_train.csv", index=False)
modded_test.to_csv("modded_test.csv", index=False)
df = pd.DataFrame(columns=['Run ID', 'Standardized', 'Normalized', 'Context Features',
                           'STD DVT context', 'Distance with X, Y', 'Tree Max Depth',
                           ])
pd.set_option('display.max_columns', None)

def remove_context_features(train_data, test_data):
    """
    Removes the features X_context and Y_context in both the training and testing data.
    Alters the file path name with the modified csv.
    :param train_data: file name of the training data
    :param test_data: file name of the testing data
    :return: New path names including the modified csv
    """
    df_train = pd.read_csv(train_data)
    df_train = df_train.drop(columns=['X_context', 'Y_context'], errors='ignore')
    df_train.to_csv("modded_train.csv", index=False)
    TRAINING_DATA = "modded_train.csv"
    df_test = pd.read_csv(test_data)
    df_test = df_test.drop(columns=['X_context', 'Y_context'], errors='ignore')
    df_test.to_csv("modded_test.csv", index=False)
    # print(f"desde el remove context, df_test: {df_test.columns}")
    TESTING_DATA = "modded_test.csv"


    return TRAINING_DATA, TESTING_DATA

def remove_std_dvt_context(train_data, test_data):
    """
    Removes the features σ(X)_context and σ(Y)_context in both the training and testing data.
    Alters the file path name with the modified csv.
    :param train_data: file name of the training data
    :param test_data: file name of the testing data
    :return: New path names including the modified csv
    """
    if train_data != "modded_train.csv":
        TRAINING_DATA = C_TRAINING_DATA
        TESTING_DATA = C_TESTING_DATA
    df_train = pd.read_csv(train_data)
    df_train = df_train.drop(columns=['σ(X)_context', 'σ(Y)_context'], errors='ignore')
    df_train.to_csv("modded_train.csv", index=False)
    TRAINING_DATA = "modded_train.csv"

    df_test = pd.read_csv(test_data)
    df_test = df_test.drop(columns=['σ(X)_context', 'σ(Y)_context'], errors='ignore')
    df_test.to_csv("modded_test.csv", index=False)
    TESTING_DATA = "modded_test.csv"
    return TRAINING_DATA, TESTING_DATA


def calc_distance_parameter(train_data, test_data):
    """
    Calculates the Euclidean distance given X_drive, X_sink, Y_drive, Y_sink. Removes
    the aforementioned, adds a new parameter called "Distance"
    :param train_data: file name of the training data
    :param test_data: file name of the testing data
    :return: New path names including the modified csv
    """

    if train_data != "modded_train.csv":
        TRAINING_DATA = C_TRAINING_DATA
        TESTING_DATA = C_TESTING_DATA
    df_train = pd.read_csv(train_data)
    df_train['Distance'] = np.sqrt((df_train['X_drive'] - df_train['X_sink']) ** 2 + (df_train['Y_drive'] - df_train['Y_sink']) ** 2)
    df_train = df_train.drop(columns=['X_drive', 'Y_drive', 'X_sink', 'Y_sink'])
    df_train.to_csv("modded_train.csv", index=False)
    TRAINING_DATA = "modded_train.csv"

    df_test = pd.read_csv(test_data)
    df_test['Distance'] = np.sqrt((df_test['X_drive'] - df_test['X_sink']) ** 2 + (df_test['Y_drive'] - df_test['Y_sink']) ** 2)
    df_test = df_test.drop(columns=['X_drive', 'Y_drive', 'X_sink', 'Y_sink'])
    df_test.to_csv("modded_test.csv", index=False)
    TESTING_DATA = "modded_test.csv"


    return TRAINING_DATA, TESTING_DATA





current_run = 0
"""         SCRIPT LOOP         """
while current_run < NUMBER_OF_RUNS:
    TRAINING_DATA = C_TRAINING_DATA
    TESTING_DATA = C_TESTING_DATA

    data_type = random.choice([0, 1, 2])  # Selects a type of adjustment to be made on the data
    # data_type = 0 → Nothing
    # data_type = 1 → Standardization
    # data_type = 2 → Normalization

    LR_type = random.choice([0,1])  # Selects the type of linear regressor to be used to fit the model
    # LR_type = 0 → LinearRegressor()
    # LR_type = 1 → Ridge(alpha = 1.0)

    tree_max_depth = random.randint(5,15) # best = 9
    tree_max_features = random.randint(5, 15) # best = 15

    # Parameter selection
    context_features =  False # random.choice([True, False])
    std_dvt_context =  False#random.choice([True, False])
    distance_parameter = True #random.choice([True, False])

    if not context_features:
        TRAINING_DATA, TESTING_DATA = remove_context_features(TRAINING_DATA, TESTING_DATA)
        # print(f"training data: {TRAINING_DATA}, testing data: {TESTING_DATA}")
        # df1 = pd.read_csv(TRAINING_DATA)
        # d21 = pd.read_csv(TESTING_DATA)
        # print(f"training df: {df1.columns}, testing df: {d21.columns}")
    if not std_dvt_context:
        # print("executed context?")
        TRAINING_DATA, TESTING_DATA = remove_std_dvt_context(TRAINING_DATA, TESTING_DATA)
    if distance_parameter :
        # print("executed distance?")
        TRAINING_DATA, TESTING_DATA = calc_distance_parameter(TRAINING_DATA, TESTING_DATA)


    """ TRAINING  """
    print(f"Executing run # {current_run} out of {NUMBER_OF_RUNS}.")
    # print(f"\tdata type: {data_type}")

    X_train, X_test, y_train, y_test = readcsv(TRAINING_DATA, data_type)
    hb = HybridModel()
    # print(f'X_train {X_train}')
    # print(f'X_test {X_test}')
    hb.fit(X_train, y_train, LR_type, [tree_max_depth, tree_max_features])
    # with open("hb_instance2.pk1", "wb") as output_file:
    #     pickle.dump(hb, output_file)

    """ TESTING """
    X_train, X_test, y_train, y_test = readcsv_p(TESTING_DATA, data_type)
    hb.predict(X_test)
    # todo: maybe a dictionary would work better here
    (large_error, small_error, ML_MAE, ML_MSE, OPL_MAE, OPL_MSE, MAE_DIFF, MSE_DIFF, R2_SCORE, ML_pcorr, ML_p_value, ML_MAE_f, ML_MSE_f,
     OPL_MAE_f, OPL_MSE_f, MAE_DIFF_f, MSE_DIFF_f, R2_SCORE_f, ML_pcorr_f,
     ML_p_value_f) = build_df_imported(hb.linear_predictions, X_test, y_test )


    """ DataFrame filling logic"""

    if data_type == 0:
        standardized, normalized = 'No', 'No'
    elif data_type == 1:
        standardized, normalized = 'Yes', 'No'
    elif data_type == 2:
        standardized, normalized = 'No', 'Yes'


    new_row = {
        'Run ID': current_run,
        'Ridge LR': LR_type,
        'Standardized': standardized,
        'Normalized': normalized,
        'Context Features': "Yes" if context_features else "No",
        'STD DVT context': "Yes" if std_dvt_context else "No",
        'Distance with X, Y': "Yes" if distance_parameter else "No",
        'Tree Max Depth': tree_max_features,
        'Biggest 4000th error': large_error,
        'Smallest 50kth error': small_error,
        'MAE linear reg': ML_MAE,
        'MSE linear reg': ML_MSE,
        'MAE OPL': OPL_MAE,
        'MSE OPL': OPL_MSE,
        'MAE diff': MAE_DIFF,
        'MSE diff': MSE_DIFF,
        'R2': R2_SCORE,
        'Pearson coeff': ML_pcorr,
        'Pearson P': ML_p_value,
        'MAE linear reg f': ML_MAE_f,
        'MSE linear reg f': ML_MSE_f,
        'MAE OPL f': OPL_MAE_f,
        'MSE OPL f': OPL_MSE_f,
        'MAE diff f': MAE_DIFF_f,
        'MSE diff f': MSE_DIFF_f,
        'R2 f': R2_SCORE_f,
        'Pearson coeff f': ML_pcorr_f,
        'Pearson P f': ML_p_value_f
    }
    new_row_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_row_df], ignore_index=True)


    print(df)
    current_run += 1

df.to_csv("output_file.csv", index=False)
df.to_excel('output_file.xlsx', index=False)









