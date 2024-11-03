"""
To run every scenario in output.xlsx, and find the best resulting model in test_labels.csv and design_csv.
"""
from data import remove_context_features, remove_std_dvt_context, calc_distance_parameter
import pandas as pd
import numpy as np
from hybridmodel import HybridModel
from train import readcsv
from predict import readcsv_p
from results_run import build_df_imported

excel_src = 'output_file.xlsx'
df = pd.read_excel(excel_src, sheet_name='Sheet1', engine='openpyxl')
# print(df.columns)

def remove_context_features(train_data, test1, test2):
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
    df_test1 = pd.read_csv(test1)
    df_test1 = df_test1.drop(columns=['X_context', 'Y_context'], errors='ignore')
    df_test1.to_csv("modded_test1.csv", index=False)
    # print(f"desde el remove context, df_test: {df_test.columns}")
    TESTING1_DATA = "modded_test1.csv"

    df_test2 = pd.read_csv(test2)
    df_test2 = df_test2.drop(columns=['X_context', 'Y_context'], errors='ignore')
    df_test2.to_csv("modded_test2.csv", index=False)
    # print(f"desde el remove context, df_test: {df_test.columns}")
    TESTING2_DATA = "modded_test2.csv"


    return TRAINING_DATA, TESTING1_DATA, TESTING2_DATA

def remove_std_dvt_context(train_data, test1, test2):
    """
    Removes the features σ(X)_context and σ(Y)_context in both the training and testing data.
    Alters the file path name with the modified csv.
    :param train_data: file name of the training data
    :param test_data: file name of the testing data
    :return: New path names including the modified csv
    """
    # if train_data != "modded_train.csv":
    #     TRAINING_DATA = C_TRAINING_DATA
    #     TESTING_DATA = C_TESTING_DATA
    df_train = pd.read_csv(train_data)
    df_train = df_train.drop(columns=['σ(X)_context', 'σ(Y)_context'], errors='ignore')
    df_train.to_csv("modded_train.csv", index=False)
    TRAINING_DATA = "modded_train.csv"

    df_test1 = pd.read_csv(test1)
    df_test1 = df_test1.drop(columns=['σ(X)_context', 'σ(Y)_context'], errors='ignore')
    df_test1.to_csv("modded_test1.csv", index=False)
    TESTING1_DATA = "modded_test1.csv"

    df_test2 = pd.read_csv(test2)
    df_test2 = df_test2.drop(columns=['σ(X)_context', 'σ(Y)_context'], errors='ignore')
    df_test2.to_csv("modded_test2.csv", index=False)
    TESTING2_DATA = "modded_test2.csv"
    return TRAINING_DATA, TESTING1_DATA, TESTING2_DATA


def calc_distance_parameter(train_data, test1, test2):
    """
    Calculates the Euclidean distance given X_drive, X_sink, Y_drive, Y_sink. Removes
    the aforementioned, adds a new parameter called "Distance"
    :param train_data: file name of the training data
    :param test_data: file name of the testing data
    :return: New path names including the modified csv
    """

    # if train_data != "modded_train.csv":
    #     TRAINING_DATA = C_TRAINING_DATA
    #     TESTING_DATA = C_TESTING_DATA
    df_train = pd.read_csv(train_data)
    df_train['Distance'] = np.sqrt((df_train['X_drive'] - df_train['X_sink']) ** 2 + (df_train['Y_drive'] - df_train['Y_sink']) ** 2)
    df_train = df_train.drop(columns=['X_drive', 'Y_drive', 'X_sink', 'Y_sink'])
    cols = df_train.columns.tolist()
    cols.remove('Distance')
    delay_idx = cols.index(' Delay')
    new_cols_order = cols[:delay_idx + 1] + ['Distance'] + cols[delay_idx + 1:]
    df_train = df_train[new_cols_order]

    df_train.to_csv("modded_train.csv", index=False)
    TRAINING_DATA = "modded_train.csv"

    df_test1 = pd.read_csv(test1)
    df_test1['Distance'] = np.sqrt((df_test1['X_drive'] - df_test1['X_sink']) ** 2 + (df_test1['Y_drive'] - df_test1['Y_sink']) ** 2)
    df_test1 = df_test1.drop(columns=['X_drive', 'Y_drive', 'X_sink', 'Y_sink'])
    cols = df_test1.columns.tolist()
    cols.remove('Distance')
    delay_idx = cols.index(' Delay')
    new_cols_order = cols[:delay_idx + 1] + ['Distance'] + cols[delay_idx + 1:]
    df_test1 = df_test1[new_cols_order]
    df_test1.to_csv("modded_test1.csv", index=False)
    TESTING1_DATA = "modded_test1.csv"

    df_test2 = pd.read_csv(test2)
    df_test2['Distance'] = np.sqrt(
        (df_test2['X_drive'] - df_test2['X_sink']) ** 2 + (df_test2['Y_drive'] - df_test2['Y_sink']) ** 2)
    df_test2 = df_test2.drop(columns=['X_drive', 'Y_drive', 'X_sink', 'Y_sink'])
    cols = df_test2.columns.tolist()
    cols.remove('Distance')
    delay_idx = cols.index(' Delay')
    new_cols_order = cols[:delay_idx + 1] + ['Distance'] + cols[delay_idx + 1:]
    df_test2 = df_test2[new_cols_order]
    df_test2.to_csv("modded_test2.csv", index=False)
    TESTING2_DATA = "modded_test2.csv"


    return TRAINING_DATA, TESTING1_DATA, TESTING2_DATA



df2 = pd.DataFrame()
for i in range(0,200):
    training_path = 'slow.csv'
    labels_path = 'labels_slow.csv'
    design_path = 'designs_slow.csv'

    modded_train = pd.read_csv(training_path)
    modded_test1 = pd.read_csv(labels_path)
    modded_test2 = pd.read_csv(design_path)
    modded_train.to_csv("modded_train.csv", index=False)
    modded_test1.to_csv("modded_test1.csv", index=False)
    modded_test2.to_csv("modded_test2.csv", index=False)

    row = df.iloc[i]
    run_id, standardized, normalized = row['Run ID'], False if row['Standardized'] == "No" else True, False if row['Normalized'] == "No" else True
    context_features, std_dvt_context, distance_parameter = False if row['Context Features'] == "No" else True, False if row['STD DVT context'] == "No" else True, False if row['Distance with X, Y'] == "No" else True
    tree_max_depth, LR_type = row['Max Tree Features'], row['Ridge LR']
    tree_max_features = row['Max Tree Features']
    training_rmse = row['RMSE linear reg']
    print(f"LR TYPE {LR_type}")

    if not standardized and not normalized:
        data_type = 0
    elif standardized and not normalized:
        data_type = 1
    elif not standardized and normalized:
        data_type = 2


    if not context_features:
        training_path, labels_path, design_path = remove_context_features(training_path, labels_path, design_path)
    if not std_dvt_context:
        # print("executed context?")
        training_path, labels_path, design_path = remove_std_dvt_context(training_path, labels_path, design_path)
    if distance_parameter:
        # print("executed distance?")
        training_path, labels_path, design_path = calc_distance_parameter(training_path, labels_path, design_path)

    """ TRAINING with slow.csv """

    X_train, X_test, y_train, y_test = readcsv(training_path, data_type)
    hb = HybridModel()
    hb.fit(X_train, y_train, LR_type, [tree_max_depth, tree_max_features])

    """ TESTING WITH LABELS """
    X_train, X_test, y_train, y_test = readcsv_p(labels_path, data_type)
    hb.predict(X_test)
    # todo: maybe a dictionary would work better here
    (large_error, small_error, ML_MAE_labels, ML_MSE_labels, OPL_MAE_labels, OPL_MSE, MAE_DIFF, MSE_DIFF, R2_SCORE_labels, ML_pcorr_labels, ML_p_value_labels,
     ML_MAE_f, ML_MSE_f,
     OPL_MAE_f, OPL_MSE_f, MAE_DIFF_f, MSE_DIFF_f, R2_SCORE_f, ML_pcorr_f,
     ML_p_value_f, rows_removed, OPL_RMSE_labels, ML_RMSE_labels, OPL_RMSE_f, ML_RMSE_f) = build_df_imported(hb.linear_predictions, X_test, y_test)

    """ TESTING WITH DESIGN """
    X_train, X_test, y_train, y_test = readcsv_p(design_path, data_type)
    hb.predict(X_test)
    # todo: maybe a dictionary would work better here
    (large_error, small_error, ML_MAE_design, ML_MSE_design, OPL_MAE_design, OPL_MSE, MAE_DIFF, MSE_DIFF, R2_SCORE_design,
     ML_pcorr_design, ML_p_value_design,
     ML_MAE_f, ML_MSE_f,
     OPL_MAE_f, OPL_MSE_f, MAE_DIFF_f, MSE_DIFF_f, R2_SCORE_f, ML_pcorr_f,
     ML_p_value_f, rows_removed, OPL_RMSE_design, ML_RMSE_design, OPL_RMSE_f, ML_RMSE_f) = build_df_imported(
        hb.linear_predictions,
        X_test, y_test)

    new_row = {
        'Run ID': run_id,
        'Ridge LR': LR_type,
        'Standardized': standardized,
        'Normalized': normalized,
        'Context Features': "Yes" if context_features else "No",
        'STD DVT context': "Yes" if std_dvt_context else "No",
        'Distance with X, Y': "Yes" if distance_parameter else "No",
        'Tree Max Depth': tree_max_depth,
        'Max Tree Features': tree_max_features,
        'RMSE linear reg': training_rmse,
        'Labels ML RMSE': ML_RMSE_labels,
        'Labels OPL RMSE': OPL_RMSE_labels,
        'Labels ML MAE': ML_MAE_labels,
        'Labels OPL MAE': OPL_MAE_labels,
        'Labels ML CORR': ML_pcorr_labels,
        'Labels ML p value': ML_p_value_labels,
        'Labels ML R2': R2_SCORE_labels,
        'Design ML RMSE': ML_RMSE_design,
        'design OPL RMSE': OPL_RMSE_design,
        'design ML MAE': ML_MAE_design,
        'design OPL MAE': OPL_MAE_design,
        'design ML CORR': ML_pcorr_design,
        'design ML p value': ML_p_value_design,
        'design ML R2': R2_SCORE_design,

    }
    new_row_df = pd.DataFrame([new_row])
    df2 = pd.concat([df2, new_row_df], ignore_index=True)
    print(df2)
    print(f"current row {row['Run ID']}")

df2.to_csv("resultados_en_tests.csv", index=False)
df2.to_excel('resultados_en_tests.xlsx', index=False)