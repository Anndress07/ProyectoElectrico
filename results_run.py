import pickle

#import predict as predict
from predict import readcsv_p

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from scipy.stats import pearsonr
from data import remove_context_features, remove_std_dvt_context, calc_distance_parameter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.metrics import precision_score


# TEST_DATA = "designs_slow.csv"
TEST_DATA = "labels_slow.csv"
def build_df_imported(pred_results, X_test, actual_results =  None):
    if actual_results is not None:
        y_test = list(actual_results)
        pred_dataframe = pred_results
        pred_dataframe["y_test"] = pred_dataframe['idx on X_test'].apply(lambda x: y_test[int(x)])
        pred_dataframe['error'] = abs(pred_dataframe['y_test'] - pred_dataframe['y_pred'])
        pd.set_option('display.float_format', '{:.3f}'.format)

        df2 = pd.DataFrame(columns=['Biggest 4000th error', 'Smallest 50kth error'])

        large_error = pred_dataframe.nlargest(4000, 'error')
        large_error = large_error.iloc[large_error.shape[0]-1]['error']
        # print(f"large error {large_error}")

        small_error = pred_dataframe.nsmallest(50000, 'error')
        small_error = small_error.iloc[small_error.shape[0]-1]['error']

        (ML_MAE, ML_MSE, OPL_MAE, OPL_MSE, MAE_DIFF, MSE_DIFF, R2_SCORE, ML_pcorr,
         ML_p_value, OPL_RMSE, ML_RMSE) = data_visualization(pred_dataframe, X_test, y_test, False)
        low_error,rows_removed = data_filtered_low_error(pred_dataframe)
        (ML_MAE_f, ML_MSE_f, OPL_MAE_f, OPL_MSE_f, MAE_DIFF_f, MSE_DIFF_f, R2_SCORE_f, ML_pcorr_f,
         ML_p_value_f, OPL_RMSE_f, ML_RMSE_f) = data_visualization(low_error, X_test, y_test, False)

        new_row = pd.DataFrame({"Biggest 4000th error" : [large_error],
                                "Smallest 50kth error": [small_error]})
        df2 = pd.concat([df2, new_row], ignore_index=True)
        # new_row = {
        #     'MAE linear reg': ML_MAE,
        #     'MSE linear reg': ML_MSE,
        #     'MAE OPL': OPL_MSE,
        #     'MSE OPL': OPL_MSE,
        #     'MAE diff': OPL_MAE - ML_MAE,
        #     'MSE diff': OPL_MSE - ML_MSE,
        #     'R2': R2_SCORE,
        #     'Pearson coeff': ML_pcorr,
        #     'Pearson P': ML_p_value
        # }
        # new_row_df = pd.DataFrame([new_row])
        # df = pd.concat([df, new_row_df], ignore_index=True)

        # results_df = pd.concat([df2, result_unfiltered], axis=1)
        # results_df = pd.concat([results_df, result_filtered], axis=1)

        return large_error, small_error, ML_MAE, ML_MSE, OPL_MAE, OPL_MSE, MAE_DIFF, MSE_DIFF, R2_SCORE, ML_pcorr, ML_p_value, ML_MAE_f, ML_MSE_f, OPL_MAE_f, OPL_MSE_f, MAE_DIFF_f, MSE_DIFF_f, R2_SCORE_f, ML_pcorr_f, ML_p_value_f, rows_removed, OPL_RMSE, ML_RMSE, OPL_RMSE_f, ML_RMSE_f
def build_df_native():
    new_data = remove_context_features(TEST_DATA)
    new_data = remove_std_dvt_context(new_data)
    new_data = calc_distance_parameter(new_data)
    X_train, X_test, y_train, y_test = readcsv_p(new_data, 0)

    with open("hb_instance2.pk1", "rb") as input_file:
        hb = pickle.load(input_file)

    y_test = list(y_test)

    pred_dataframe = hb.linear_predictions
    pred_dataframe["y_test"] = pred_dataframe['idx on X_test'].apply(lambda x: y_test[int(x)])
    pred_dataframe['error'] = abs(pred_dataframe['y_test'] - pred_dataframe['y_pred'])
    pd.set_option('display.float_format', '{:.3f}'.format)
    design_column = pd.read_csv('design_column.csv')
    pred_dataframe['Design'] = design_column

    # print(pred_dataframe)
    # y_pred = pred_dataframe['y_pred']
    pred_dataframe = pred_dataframe[pred_dataframe['Design'] == 's15850']
    print("====================")
    print(f"pred dataframe {pred_dataframe}")

    return pred_dataframe, X_test, y_test

def first_data(pred_dataframe):
    y_pred = pred_dataframe['y_pred']
    large_error = pred_dataframe.nlargest(4000, 'error')
    small_error = pred_dataframe.nsmallest(50000, 'error')
    print(f"largest errors:\n {large_error}")
    print(f"smallest errors:\n {small_error}")

    # rows_to_display = pred_dataframe.iloc[[23960, 25870, 56097, 42310, 15001]]
    # print(rows_to_display)


def data_filtered_low_error(pred_dataframe):
    new_df = pred_dataframe.copy()
    indices_to_remove = new_df[new_df['error'] > 10]['idx on X_test'].tolist()
    for idx in indices_to_remove:
        if idx in new_df['y_pred']:
            new_df['y_pred'].drop(idx)

    rows_before = len(new_df)
    new_df.drop(new_df[new_df['error'] > 10].index, inplace=True)
    rows_after = len(new_df)
    rows_removed = rows_before - rows_after
    print(f"Number of instances removed: {rows_removed}")

    return new_df, rows_removed


def data_visualization(pred_dataframe, X_test, y_test, plots_enable):
    # print("Original results of the model")
    # print("\tMAE test", mean_absolute_error(pred_dataframe['y_test'], pred_dataframe['y_pred']))
    # print("\tMean Squared Error (MSE) test:", mean_squared_error(pred_dataframe['y_test'], pred_dataframe['y_pred']))
    # print("\tR-squared Score test: ", r2_score(pred_dataframe['y_test'], pred_dataframe['y_pred']))

    # print("Results for the model with low error")
    # print("\tMAE test", mean_absolute_error(pred_dataframe['y_test'], pred_dataframe['y_pred']))
    # print("\tMean Squared Error (MSE) test:", mean_squared_error(pred_dataframe['y_test'], pred_dataframe['y_pred']))
    # print("\tR-squared Score test: ", r2_score(pred_dataframe['y_test'], pred_dataframe['y_pred']))
    ML_MAE = mean_absolute_error(pred_dataframe['y_test'], pred_dataframe['y_pred'])
    ML_MSE = mean_squared_error(pred_dataframe['y_test'], pred_dataframe['y_pred'])
    OPL_MAE = mean_absolute_error(pred_dataframe['y_test'], pred_dataframe['opl_pred'])
    OPL_MSE = mean_squared_error(pred_dataframe['y_test'], pred_dataframe['opl_pred'])

    R2_SCORE = r2_score(pred_dataframe['y_test'], pred_dataframe['y_pred'])

    ML_pcorr, ML_p_value = pearsonr(pred_dataframe['y_test'], pred_dataframe['y_pred'])
    OPL_pcorr, OPL_p_value = pearsonr(pred_dataframe['y_test'], pred_dataframe['opl_pred'])

    # OPL_delay = X_test[' Delay']
    # ML_MAE = mean_absolute_error(pred_dataframe['y_test'], pred_dataframe['y_pred'])
    OPL_RMSE = root_mean_squared_error(pred_dataframe['y_test'], pred_dataframe['opl_pred'])
    ML_RMSE = root_mean_squared_error(pred_dataframe['y_test'], pred_dataframe['y_pred'])
    MAE_DIFF = OPL_MAE - ML_MAE
    MSE_DIFF = OPL_MSE - ML_MSE
    # OPL_pcorr, OPL_p_value = pearsonr(pred_dataframe['y_test'], OPL_delay)
    print(f"ML RMSE {ML_RMSE}, OPL RMSE {OPL_RMSE}")
    print(f"ML MAE {ML_MAE}, OPL MAE {OPL_MAE}")
    print(f"ML CORR {ML_pcorr} ML P value {ML_p_value}, OPL CORR {OPL_pcorr} OPL P value {OPL_p_value}")
    print(f"R2 SCORE {R2_SCORE}")


    # print(f"\tOPL_RMSE: {OPL_RMSE}")
    # print(f"\tML_RMSE: {ML_RMSE}")

    # print(f"\tOPL Pearson coeff: {OPL_pcorr}")
    # print(f"\tOPL Pearson p value: {OPL_p_value}")

    # print(f"\tML Pearson coeff: {ML_pcorr}")
    # print(f"\tML Pearson p value: {ML_p_value}")

    if plots_enable:
        plt.scatter(pred_dataframe['y_test'], pred_dataframe['y_pred'], color='blue', label='Predictions')
        plt.plot([min(pred_dataframe['y_test']), max(pred_dataframe['y_test'])], [min(pred_dataframe['y_pred']), max(pred_dataframe['y_pred'])], color='red', label='Ideal Fit')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('True vs Predicted Values')
        plt.legend()
        plt.show()


        # plt.hist(pred_dataframe[])
        fig, axs = plt.subplots(1,2, figsize=(10,5))
        axs[0].hist(pred_dataframe["y_pred"], bins=500, color='blue', edgecolor='black')
        axs[0].set_title('Histogram of y_pred')
        axs[0].set_xlabel('Values')
        axs[0].set_ylabel('Frequency')

        # Histogram for data2
        axs[1].hist(pred_dataframe["error"], bins=500, color='red', edgecolor='black')
        axs[1].set_title('Histogram for the error')
        axs[1].set_xlabel('Values')
        axs[1].set_ylabel('Frequency')

        plt.tight_layout()  # Automatically adjust subplot parameters to give padding
        plt.show()
        plt.clf()

    df = pd.DataFrame(columns=['MAE linear reg', 'MSE linear reg', 'MAE OPL', 'MSE OPL',
'MAE diff', 'MSE diff', 'R2', 'Pearson coeff', 'Pearson P'])

    new_row = {
        'MAE linear reg': ML_MAE,
        'MSE linear reg': ML_MSE,
        'MAE OPL': OPL_MAE,
        'MSE OPL': OPL_MSE,
        'MAE diff':MAE_DIFF ,
        'MSE diff': MSE_DIFF ,
        'R2': R2_SCORE,
        'Pearson coeff': ML_pcorr,
        'Pearson P': ML_p_value
    }
    new_row_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_row_df], ignore_index=True)

    return ML_MAE, ML_MSE, OPL_MAE, OPL_MSE, MAE_DIFF, MSE_DIFF, R2_SCORE, ML_pcorr, ML_p_value, OPL_RMSE, ML_RMSE



if __name__ == "__main__":
    results_df, X_test, y_test = build_df_native()
    # filtered_df = data_filtered_low_error(results_df)
    first_data(results_df)
    print(f"Results for all predictions")
    data_visualization(results_df, X_test, y_test, True )
    print(f"Results for predictions with acceptable error")
    # data_visualization(filtered_df, X_test, y_test, True)
