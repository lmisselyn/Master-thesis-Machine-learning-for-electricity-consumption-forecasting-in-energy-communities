from datetime import datetime, timedelta
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from helper import *
import warnings

def mlp_model(set, scale=False, show=False):
    warnings.filterwarnings('ignore')
    """
    train a random multi layer perceptron model with the dataset 'filename'
    - set (optional) : provide train and test sets
    - scale (boolean) : scale data if true
    """
    x_train = set[0]
    y_train = set[1]
    x_test = set[2]
    y_test = set[3]

    if scale:
        sc = StandardScaler()
        scaler = sc.fit(x_train)
        x_train = pd.DataFrame(scaler.transform(x_train), x_train.index, x_train.columns)
        x_test = pd.DataFrame(scaler.transform(x_test), x_test.index, x_train.columns)

    model = MLPRegressor(
        hidden_layer_sizes=(100, 100, 100),
        activation='relu',
        solver='adam',
        learning_rate='adaptive',
        learning_rate_init=0.005,
        max_iter=1500,
        early_stopping=True,
        validation_fraction=0.1)

    model.fit(x_train, y_train)

    if show:
        y_predict = model.predict(x_test)
        aggregated = aggregate(y_test.values, y_predict)
        plot_model(y_test.values, y_predict, 'mlp')
        plot_model(aggregated[0], aggregated[1], 'mlp_aggregated')
        print('Accuracy :')
        print(evaluate_model(y_test.values, y_predict))
        print("Aggregated accuracy :")
        print(evaluate_model(aggregated[0], aggregated[1]))

    return model

def parameter_search():

    parameters = {'hidden_layer_sizes': [(100, 200), (100, 100, 100), (100, 200, 500), (250, 500), (250, 500, 1000), (64, 128, 64)],
                  'activation': ['relu'],
                  'solver': ['adam'],
                  'learning_rate': ['adaptive'],
                  'learning_rate_init': [0.005],
                  'max_iter': [1500],
                  'shuffle': [False],
                  'warm_start': [False],
                  'early_stopping': [True]}

    var10 = ['Minutes', 'Snow depth', 'Day', 'Weekend', 'Snowfall']

    df = pd.read_csv('../Datasets/10_test.csv', index_col='Datetime')
    df = df['2020-02-16 00:00:00':'2020-08-16 00:00:00']

    df.reset_index(inplace=True)
    x_train = df[var10]
    y_train = df["Consumption(Wh)"]

    tscv = TimeSeriesSplit(n_splits=5, test_size=672)

    mlp_gs = GridSearchCV(MLPRegressor(), param_grid=parameters, cv=tscv,
                          scoring='neg_root_mean_squared_error')
    mlp_gs.fit(x_train, y_train)
    best_params = mlp_gs.best_params_
    best_score = mlp_gs.best_score_

    with open('mlp_gridsearch', 'w') as f:
        f.write(str(best_params) + '\n' + str(best_score))


if __name__ == '__main__':

    variables10 = ['Minutes', 'Snow depth', 'Day', 'Weekend', 'Snowfall']
    var10 = ['Previous_4d_mean_cons', 'Week', 'Snowfall', 'Day', 'Weekend']

    df = pd.read_csv('../Datasets/10_test.csv', index_col=["Datetime"],
                             parse_dates=["Datetime"])

    #date = datetime.fromisoformat()
    x = df[var10]
    y = df["Consumption(Wh)"]
    x_train = x['2020-02-16 00:00:00':'2021-01-07 00:00:00']
    y_train = y['2020-02-16 00:00:00':'2021-01-07 00:00:00']
    x_test = x['2021-01-07 00:00:00':'2021-01-08 00:00:00']
    y_test = y['2021-01-07 00:00:00':'2021-01-08 00:00:00']

    mlp_model(set=[x_train, y_train, x_test, y_test], show=True, scale=True)

    #parameter_search()