from datetime import datetime, timedelta

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


def mlp_model(set, scale=False, show=False):
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
        hidden_layer_sizes=(250, 500,),
        activation='relu',
        solver='adam',
        alpha=0.0005,
        batch_size='auto',
        learning_rate='adaptive',
        learning_rate_init=0.007,
        power_t=0.5,
        max_iter=500,
        random_state=None,
        tol=0.00005,
        verbose=0,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        n_iter_no_change=10,
        max_fun=1500)

    model.fit(x_train, y_train)
    """
    if show:
        y_predict = model.predict(x_test)
        aggregated = helper.aggregate(y_test.values, y_predict)
        helper.plot_model(y_test.values, y_predict, 'mlp')
        helper.plot_model(aggregated[0], aggregated[1], 'mlp_aggregated')
        print('Accuracy :')
        print(helper.evaluate_model(y_test.values, y_predict))
        print("Aggregated accuracy :")
        print(helper.evaluate_model(aggregated[0], aggregated[1]))
    """
    return model

def parameter_search():
    parameters = {'hidden_layer_sizes': [(100, 200), (100, 100, 100), (100, 200, 500), (250, 500), (250, 500, 1000), (64, 128, 64)],
                  'activation': ['relu', 'tahn', 'logistic'],
                  'solver': ['sgd', 'adam'],
                  'batch_size': [200, 300, 400, 500],
                  'learning_rate': ['constant', 'adaptive'],
                  'learning_rate_init': [0.001, 0.002, 0.003, 0.005],
                  'max_iter': [200, 300, 500, 1000],
                  'shuffle': [False, True],
                  'warm_start': [False, True],
                  'early_stopping': [False, True]}

    var10 = ['Previous_4d_mean_cons', 'Snow depth', 'Weekend', 'Irradiation', 'Minutes', 'Week',
             'Wind direction', 'Month', 'Snowfall', 'Temperature', 'Rainfall']

    df = pd.read_csv('../Datasets/10_test.csv', index_col='Datetime')
    df = df['2020-02-16 00:00:00':'2020-08-16 00:00:00']

    df.reset_index(inplace=True)
    x_train = df[var10]
    y_train = df["Consumption(Wh)"]

    sc = StandardScaler()
    scaler = sc.fit(x_train)
    x_train = scaler.transform(x_train)

    tscv = TimeSeriesSplit(n_splits=5, test_size=672)

    mlp_gs = GridSearchCV(MLPRegressor(), param_grid=parameters, cv=tscv,
                          scoring='neg_root_mean_squared_error')
    mlp_gs.fit(x_train, y_train)
    best_params = mlp_gs.best_params_
    best_score = mlp_gs.best_score_

    with open('mlp_gridsearch', 'w') as f:
        f.write(str(best_params) + '\n' + str(best_score))


if __name__ == '__main__':
    parameter_search()