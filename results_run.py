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

def build_df_imported(pred_results, actual_results =  None):
    if actual_results is not None:
        # y_test = list(actual_results)
        opl_pred_column = pd.read_csv('opl_delay_column.csv')
        pred_dataframe = pred_results
        pred_dataframe['opl_pred'] = opl_pred_column
        pred_dataframe["y_test"] = pred_dataframe['idx on X_test'].apply(lambda x: actual_results[int(x)])
        pred_dataframe['error'] = pred_dataframe['y_test'] - pred_dataframe['y_pred']
        pd.set_option('display.float_format', '{:.3f}'.format)

    return pred_dataframe


def generate_metrics(pred_dataframe, plots_enable):
    ML_MAE = mean_absolute_error(pred_dataframe['y_test'], pred_dataframe['y_pred'])
    ML_MSE = mean_squared_error(pred_dataframe['y_test'], pred_dataframe['y_pred'])
    OPL_MAE = mean_absolute_error(pred_dataframe['y_test'], pred_dataframe['opl_pred'])
    OPL_MSE = mean_squared_error(pred_dataframe['y_test'], pred_dataframe['opl_pred'])
    R2_SCORE = r2_score(pred_dataframe['y_test'], pred_dataframe['y_pred'])
    ML_pcorr, ML_p_value = pearsonr(pred_dataframe['y_test'], pred_dataframe['y_pred'])
    OPL_pcorr, OPL_p_value = pearsonr(pred_dataframe['y_test'], pred_dataframe['opl_pred'])
    OPL_RMSE = root_mean_squared_error(pred_dataframe['y_test'], pred_dataframe['opl_pred'])
    ML_RMSE = root_mean_squared_error(pred_dataframe['y_test'], pred_dataframe['y_pred'])
    MAE_DIFF = OPL_MAE - ML_MAE
    MSE_DIFF = OPL_MSE - ML_MSE
    if plots_enable:
        plt.scatter(pred_dataframe['y_test'], pred_dataframe['y_pred'], color='mediumseagreen',
                    label='Model predictions')
        plt.plot([min(pred_dataframe['y_test']), max(pred_dataframe['y_test'])],
                 [min(pred_dataframe['y_pred']), max(pred_dataframe['y_pred'])], color='dimgray', label='Ideal Fit',
                 linewidth=2)
        plt.xlabel('Post routing result')
        plt.ylabel('Hybrid model result')
        plt.title('True vs Predicted Values')
        plt.legend()
        plt.show()

        plt.scatter(pred_dataframe['y_test'], pred_dataframe['opl_pred'], color='green',
                    label='OpenLane predictions')
        plt.plot([min(pred_dataframe['y_test']), max(pred_dataframe['y_test'])],
                 [min(pred_dataframe['opl_pred']), max(pred_dataframe['opl_pred'])], color='dimgray', label='Ideal Fit',
                 linewidth=2)
        plt.xlabel('Post routing result')
        plt.ylabel('OpenLane result')
        plt.title('True vs OpenLane Predicted Values')
        plt.legend()
        plt.show()

        opl_error = pred_dataframe["y_test"] - pred_dataframe['opl_pred']
        plt.hist(pred_dataframe["error"], bins=100, color='mediumseagreen', alpha=0.7, label='Model error')
        plt.hist(opl_error, bins=100, color='dimgray', alpha=0.5, label='OpenLane error')

        # Add labels and title
        plt.xlabel('Predicted Error (ns)')
        plt.ylabel('Number of Samples')
        plt.title('Comparison of Predicted Errors')
        plt.legend()

        # Show plot
        plt.show()
    metrics_dict = {
        'ML_MAE': ML_MAE,
        'ML_MSE': ML_MSE,
        'OPL_MAE': OPL_MAE,
        'OPL_MSE': OPL_MSE,
        'MAE_DIFF': MAE_DIFF,
        'MSE_DIFF': MSE_DIFF,
        'R2_SCORE': R2_SCORE,
        'ML_pcorr': ML_pcorr,
        'ML_p_value': ML_p_value,
        'OPL_RMSE': OPL_RMSE,
        'ML_RMSE': ML_RMSE
    }
    print(metrics_dict)
    return metrics_dict

def results_method(y_test, plots_enable):
    with open("hb_instance2.pk1", "rb") as input_file:
        hb = pickle.load(input_file)

    y_test = list(y_test)

    pred_dataframe = hb.linear_predictions
    pred_dataframe = build_df_imported(pred_dataframe, y_test)
    generate_metrics(pred_dataframe, plots_enable)


# if __name__ == "__main__":
#     break
