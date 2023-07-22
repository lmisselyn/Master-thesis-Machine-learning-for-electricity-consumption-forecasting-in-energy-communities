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

source_models = {'RF': RandomForestRegressor(), 'SVM': SVR(), 'XGB': XGBRegressor(), 'MLP': MLPRegressor(),
                 'KNN': KNeighborsRegressor()}
first_d = {'01': '2020-02-25 00:00:00', '02': '2020-02-15 00:00:00', '03': '2020-02-27 00:00:00',
           '04': '2020-07-24 00:00:00',
           '05': '2020-08-22 00:00:00', '06': '2020-08-25 00:00:00', '07': '2020-08-25 00:00:00',
           '08': '2020-10-06 00:00:00'}


def parameter_search(df, parameters, model, features):
    df = df['2020-02-08 00:00:00':]

    df.reset_index(inplace=True)
    x_train = df[features]
    y_train = df["Consumption(Wh)"]

    tscv = TimeSeriesSplit(n_splits=5, test_size=672)

    mlp_gs = GridSearchCV(source_models[model], param_grid=parameters, cv=tscv,
                          scoring='neg_root_mean_squared_error')
    mlp_gs.fit(x_train, y_train)
    best_params = mlp_gs.best_params_
    best_score = mlp_gs.best_score_

    with open('Parameters/' + model + '01' + '.txt', 'w') as f:
        f.write(str(best_params) + '\n' + str(best_score))


if __name__ == '__main__':

    var = ['Day', 'Minutes', 'Week'
                                  'Weekend', 'temperature_2m', 'relativehumidity_2m',
                'dewpoint_2m', 'apparent_temperature',
                'shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
                'direct_normal_irradiance', 'windspeed_10m',
                'Prev_4d_mean_cons',
                'Prev_4w_mean_cons']  # 'winddirection_10m', 'pressure_msl', 'surface_pressure', 'precipitation', 'snowfall', 'weathercode', 'cloudcover', 'cloudcover_low', 'cloudcover_mid', 'cloudcover_high', 'Consumption(Wh)', 'Week', 'Month', 'Day_of_year',

    rf_param = {'n_estimators': [75, 100, 150, 200],
                'criterion': ['squared_error', 'absolute_error'],
                'max_depth': [None, 6, 20]}

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
                 'eta': [0.015, 0.05, 0.2, 0.3],
                 'gamma': [0, 1, 2],
                 'subsample': [0.7, 1],
                 'eval_metric': ['rmse'],
                 'early_stopping_rounds': [100],
                 'objective': ['reg:squarederror'],
                 'max_depth': [None, 5, 6],
                 'n_estimators': [100]}

    svm_param = {'C': [1, 1.5, 2, 2.5, 3],
                 'epsilon': [0.05, 0.1, 0.15, 0.2]}

    knn_param = {'n_neighbors': [100, 60, 500, 1125, 1150],
                 'weights': ['uniform'],
                 'algorithm': ['brute'],
                 'metric': ['minkowski']}

    for i in ['02']:  # , '02', '03', '04', '05', '06', '07', '08']:
        filename = 'Datasets/' + i + '/' + i + 'final.csv'
        df = pd.read_csv(filename, index_col='Datetime')
        train_first_date = datetime.datetime.fromisoformat(first_d[i])
        train_last_date = train_first_date + datetime.timedelta(weeks=16)
        features = evaluation.spearman[i]
        parameter_search(df[str(train_first_date):str(train_last_date)], knn_param, 'KNN', features)

    #parameter_search(df['2020-02-08 00:00:00':], rf_param, 'R_F', '10_test')

    #parameter_search(df['2020-02-08 00:00:00':], mlp_param, 'MLP', '10_test')

    #parameter_search(df['2020-02-08 00:00:00':], xgb_param, 'XGB', '10_test')
