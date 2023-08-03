import datetime

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

import evaluation
import global_variables
from helperv2 import get_feature

source_models = {'RF': RandomForestRegressor(), 'SVM': SVR(), 'XGB': XGBRegressor(), 'KNN': KNeighborsRegressor()}
first_d = {'01': '2020-02-25 00:00:00', '02': '2020-02-15 00:00:00', '03': '2020-02-27 00:00:00',
           '04': '2020-07-24 00:00:00',
           '05': '2020-08-22 00:00:00', '06': '2020-08-25 00:00:00', '07': '2020-08-25 00:00:00',
           '08': '2020-10-06 00:00:00'}

param = {'RF': {'n_estimators': [75, 100, 150, 200],
                'criterion': ['absolute_error'],
                'max_depth': [None, 6, 7],
                'min_samples_split': [2],
                'max_features': [1, 2],
                'max_leaf_nodes': [None],
                'random_state': [5],
                'min_impurity_decrease': [0.0],
                'bootstrap': [False],
                'n_jobs': [2],
                'max_samples': [None]},
         'XGB': {'booster': ['gbtree'],
                 'eta': [0.01, 0.0075, 0.008],
                 'eval_metric': ['rmse'],
                 'objective': ['reg:squarederror'],
                 'max_depth': [None, 6, 7],
                 'n_estimators': [100, 125, 150],
                 'colsample_bylevel': [0.2]},
         'SVM': {'C': [1, 5, 10],
                 'epsilon': [0.1, 0.2, 0.4],
                 'gamma': ['scale', 0.01, 0.001]},
         'MLP': {'hidden_layer_sizes': [(150, 150, 150), (100, 100, 100), (500, 500, 500), (1000, 100)],
                 'activation': ['relu'],
                 'solver': ['adam'],
                 'learning_rate': ['adaptive'],
                 'learning_rate_init': [0.005],
                 'max_iter': [1500],
                 'shuffle': [False],
                 'warm_start': [False],
                 'early_stopping': [True]},
         'KNN': {'n_neighbors': [5, 10, 20, 30, 50],
                 'weights': ['uniform'],
                 'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                 'metric': ['minkowski']}
         }


def parameter_search(df, parameters, model, features, dataset):
    df.reset_index(inplace=True)
    x_train = df[features]
    y_train = df["Consumption(Wh)"]

    tscv = TimeSeriesSplit(n_splits=10, max_train_size=10752, test_size=672)

    mlp_gs = GridSearchCV(source_models[model], param_grid=parameters, cv=tscv,
                          scoring='neg_mean_absolute_percentage_error')
    mlp_gs.fit(x_train, y_train)
    best_params = mlp_gs.best_params_
    best_score = mlp_gs.best_score_

    with open('Parameters/' + model + dataset + '.txt', 'w') as f:
        f.write(str(best_params) + '\n' + str(best_score))


if __name__ == '__main__':

    for i in ['01']: #, '02', '03', '04', '05', '06', '07', '08']:
        m = 'KNN'
        filename = 'Datasets/' + i + '/' + i + 'final.csv'
        df = pd.read_csv(filename, index_col='Datetime')
        train_first_date = datetime.datetime.fromisoformat(first_d[i])
        features = global_variables.pearson[i]
        parameter_search(df[str(train_first_date):], param[m], m, features, i)
