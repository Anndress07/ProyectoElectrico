# Hybrid Learning Model (Decision Tree + Linear Regression)

### A hybrid implementation of Scikit's Decision Tree and Linear Regressors intended to predict delays in digital integrated circuits. 

This project was developed as a graduation project for the class IE-0499 at the Universidad de Costa Rica, on behalf of the  [Microelectronics and Computer Architecture Research Lab (LIMA)](https://eie.ucr.ac.cr/laboratorios/lima/). The purpose of this implementation is to combine two widely known machine learning techniques—decision trees and linear 
regression—to create a fast and lightweight model for predicting net delays in integrated circuits.

This project heavily relies on the open source, RTL-to-GDSII tool [OpenLane](https://github.com/The-OpenROAD-Project/OpenLane), and it aims to improve pre-routing delay approximations provided by the tool. 

## Usage

To use the model, simply access the `main.py` and run it. The model's parameters can be modified by editing lines `9-25` of the main file. You will need data to train and test the model



```python
TRAINING_DATA = 'slow.csv'
TRAINING_SIZE = 0.8
MAX_TREE_DEPTH = 13
MAX_TREE_FEATURES = 13


TESTING_DATA = 'labels_slow.csv'
TESTING_SIZE = 1.0

context_features = False
std_dvt_context = False
distance_parameter = True

LR_type = 1
data_scaling = 0

plots_enable = True
```
<details>
  <summary style="font-size: 30px; font-weight: bold;"> Click here to view a detailed explanation of each parameter </summary>

* `TRAINING_DATA:` The file path for the CSV used to train the model. 
* `TRAINING_SIZE:` Represents the percentage of `TRAINING_DATA` to be used as training data.
* `MAX_TREE_DEPTH:` Limits the depth of the decision tree.
* `MAX_TREE_FEATURES:` Limits the feature usage of the decision tree.
* `TESTING_DATA:` The file path for the CSV used to test the model.
* `TESTING_SIZE:` Represents the percentage of `TESTING_DATA` to be used as testing data.
* `context_features`: Denotes usage of features related to the locations of the context sinks in the model.
  *  `True:` Context features are used.
  * `False:` Context features are not used.
* `std_dvt_context:` Denotes usage of features related to the standard deviation of the location of the context sinks in the model.
  *  `True:` The standard deviation of the context features are used.
  * `False:` The standard deviation of the context features are not used.
* `distance_parameter:` Denotes usage of a parameter that replaces X and Y locations of driver and sink gates with the euclidean distance between gates.
  *  `True:` The distance parameter is used.
  * `False:` The distance parameter is not used.
* `LR_type:` Represents the type of linear regressor to be used in the model.
  * `0:` The standard linear regression OLS is used.
  * `1:` The Ridge linear regression is used.
*`data_scaling:` Represents the type of data scaling to be applied to the model.
  * `0:` No data scaling is applied.
  * `1:` Standardization is applied to the data.
  * `2:` Normalization is applied to the data. 

</details>

## About the data
Datasets in the form of CSV files are needed to train and to test the model. The CSV file is formatted as

<div align="center">
  
| Fanout   | Slew     | Delay    | Distance | ...      | Label Delay|
|----------|----------|----------|----------|----------|----------  |
| 0.0      |     0.02 |    0.19  |    0.01  | ...      | 0.31       | 

</div>

Where `Label Delay` is the target variable and must be the last column of the dataframe.



