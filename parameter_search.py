import datetime

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

import evaluation
from helperv2 import get_feature

source_models = {'RF': RandomForestRegressor(), 'SVM': SVR(), 'XGB': XGBRegressor()}
first_d = {'01': '2020-02-25 00:00:00', '02': '2020-02-15 00:00:00', '03': '2020-02-27 00:00:00',
           '04': '2020-07-24 00:00:00',
           '05': '2020-08-22 00:00:00', '06': '2020-08-25 00:00:00', '07': '2020-08-25 00:00:00',
           '08': '2020-10-06 00:00:00'}


def parameter_search(df, parameters, model, features, dataset):
    df = df['2020-02-08 00:00:00':]

    df.reset_index(inplace=True)
    x_train = df[features]
    y_train = df["Consumption(Wh)"]

    tscv = TimeSeriesSplit(n_splits=6, test_size=672)

    mlp_gs = GridSearchCV(source_models[model], param_grid=parameters, cv=tscv,
                          scoring='neg_mean_absolute_percentage_error')
    mlp_gs.fit(x_train, y_train)
    best_params = mlp_gs.best_params_
    best_score = mlp_gs.best_score_

    with open('Parameters/' + model + dataset + '.txt', 'w') as f:
        f.write(str(best_params) + '\n' + str(best_score))


if __name__ == '__main__':

    var = ['Day', 'Minutes', 'Week'
                             'Weekend', 'temperature_2m', 'relativehumidity_2m',
           'dewpoint_2m', 'apparent_temperature',
           'shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
           'direct_normal_irradiance', 'windspeed_10m',
           'Prev_4d_mean_cons',
           'Prev_4w_mean_cons']  # 'winddirection_10m', 'pressure_msl', 'surface_pressure', 'precipitation', 'snowfall', 'weathercode', 'cloudcover', 'cloudcover_low', 'cloudcover_mid', 'cloudcover_high', 'Consumption(Wh)', 'Week', 'Month', 'Day_of_year',

    param = {'RF': {'n_estimators': [75, 100, 150, 200],
                    'criterion': ['absolute_error'],
                    'max_depth': [None, 6, 7],
                    'min_samples_split': [2],
                    'max_features': [1,2],
                    'max_leaf_nodes': [None],
                    'random_state': [5],
                    'min_impurity_decrease': [0.0],
                    'bootstrap' : [False],
                    'n_jobs': [2],
                    'max_samples': [None]},
             'XGB': {'booster': ['gbtree'],
                     'eta': [0.01, 0.0075, 0.008],
                     'eval_metric': ['rmse'],
                     'objective': ['reg:squarederror'],
                     'max_depth': [None, 6, 7],
                     'n_estimators': [100, 125, 150],
                     'colsample_bylevel': [0.2]},
             'SVM': {'C': [0.1, 1, 10, 100],
                     'epsilon': [0.1, 0.4],
                     'gamma': ['scale', 1, 0.1, 0.01, 0.001]}
             }

    mlp_param = {'hidden_layer_sizes': [(150, 150, 150), (100, 100, 100), (500, 500, 500), (1000, 100)],
                 'activation': ['relu'],
                 'solver': ['adam'],
                 'learning_rate': ['adaptive'],
                 'learning_rate_init': [0.005],
                 'max_iter': [1500],
                 'shuffle': [False],
                 'warm_start': [False],
                 'early_stopping': [True]}

    xgb_param = {'booster': ['gbtree'],
                 'eta': [0.01, 0.0075, 0.008],
                 'eval_metric': ['rmse'],
                 # 'early_stopping_rounds': [10, 20],
                 'objective': ['reg:squarederror'],
                 'max_depth': [None, 6],
                 'n_estimators': [100, 125, 150],
                 'colsample_bylevel': [0.2]}

    svm_param = {'C': [0.1, 1, 10, 100],
                 'epsilon': [0.1, 0.4],
                 'gamma': ['scale', 1, 0.1, 0.01, 0.001]}

    knn_param = {'n_neighbors': [100, 60, 500, 1125, 1150],
                 'weights': ['uniform'],
                 'algorithm': ['brute'],
                 'metric': ['minkowski']}

    for i in ['01', '02', '03', '04', '05', '06', '07', '08']:
        for m in source_models.keys():
            filename = 'Datasets/' + i + '/' + i + 'final.csv'
            df = pd.read_csv(filename, index_col='Datetime')
            train_first_date = datetime.datetime.fromisoformat(first_d[i])
            # train_last_date = train_first_date + datetime.timedelta(weeks=32)
            features = evaluation.pearson[i]
            parameter_search(df[str(train_first_date):], param[m], m, features, i)

    # parameter_search(df['2020-02-08 00:00:00':], rf_param, 'R_F', '10_test')

    # parameter_search(df['2020-02-08 00:00:00':], mlp_param, 'MLP', '10_test')

    # parameter_search(df['2020-02-08 00:00:00':], xgb_param, 'XGB', '10_test')
