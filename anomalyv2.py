import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

import global_variables
import helper


def new_parameters_search(df, model):
    x = df[global_variables.pearson['01']]
    y = df['Consumption(Wh)']
    n_split = 2

    parameters = {'booster': ['gbtree'],
                  'validate_parameters': [False],
                  'verbosity': [0],
                  'eval_metric': ['rmse'],
                  'early_stopping': [False],
                  'objective': ['reg:squarederror'],
                  'learning_rate': [0.006, 0.0075, 0.0085],
                  'max_depth': [1, 3, 4],
                  'max_features': [0.2, 0.5],
                  'n_estimators': [95, 125, 170],
                  'colsample_bylevel': [0.2, 0.5],
                  'subsample': [None, 0.85]}

    tscv = TimeSeriesSplit(n_splits=n_split, max_train_size=10752, test_size=672)
    xgb_gs = GridSearchCV(xgboost.XGBRegressor(), param_grid=parameters, cv=tscv,
                          scoring='neg_mean_absolute_percentage_error')

    xgb_gs.fit(x, y)
    best_params = xgb_gs.best_params_

    best_xgb_model = xgboost.XGBRegressor(**best_params)

    best_xgb_model.fit(x, y)
    return best_xgb_model

def anomaly_simulator(dataset):
    total_error = []
    #filename = 'Datasets/' + dataset + '/' + dataset + 'final.csv'
    df = pd.read_csv(dataset, index_col='Datetime')
    last_date = datetime.fromisoformat(df.index[-1])
    train_start_date = datetime.fromisoformat(global_variables.first_d['07'])
    train_end_date = train_start_date + timedelta(weeks=16)
    test_end_date = train_end_date + timedelta(weeks=1)

    features = global_variables.pearson['07']
    model = global_variables.best_model['07']

    x = df[features]
    y = df['Consumption(Wh)']
    cnt = 0
    while test_end_date < last_date:
        x_train = x[str(train_start_date):str(train_end_date)]
        y_train = y[str(train_start_date):str(train_end_date)]
        x_test = x[str(train_end_date):str(test_end_date)]
        y_test = y[str(train_end_date):str(test_end_date)]
        model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)
        y_pred = model.predict(x_test)
        aggregated = helper.aggregate(y_test.values, y_pred)
        #helper.plot_model(aggregated[0], aggregated[1], 'ok')
        MAPE = mean_absolute_percentage_error(aggregated[0], aggregated[1])
        total_error.append(MAPE)

        if MAPE > 0.4:
            print('Anomaly detected : ' + str(MAPE))
            cnt += 1
        else:
            cnt = 0
            print("MAPE : " + str(MAPE))

        if cnt == 3:
            train_start_date = test_end_date - timedelta(weeks=6)
            train_end_date = test_end_date
            test_end_date = test_end_date + timedelta(weeks=1)
            model = new_parameters_search(df[str(train_start_date):str(test_end_date)], model)
            print(model.get_params)
            #model = global_variables.best_model['08']
            cnt = 0
        else:
            train_start_date = test_end_date - timedelta(weeks=16)
            train_end_date = test_end_date
            test_end_date = test_end_date + timedelta(weeks=1)

    print("Average MAPE : " + str(np.mean(total_error)))


if __name__ == '__main__':
    #anomaly_simulator('07')
    anomaly_simulator('Datasets/anomaly_test/anomaly_test.csv')
