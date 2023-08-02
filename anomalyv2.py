from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

import global_variables
import helper


def best_model_search(dataset, end_date, model):
    filename = 'Datasets/' + dataset + '/' + i + 'final.csv'
    df = pd.read_csv(filename, index_col='Datetime')
    df = df[global_variables.first_d[dataset]: end_date]
    x = df[global_variables.pearson[dataset]]
    y = df['Consumption(Wh)']
    n_split = len(df) // 11424
    if n_split == 1:
        n_split = 2

    parameters = {'booster': ['gbtree'],
                  'validate_parameters': [False],
                  'verbosity': [0],
                  'eval_metric': ['rmse'],
                  'early_stopping': [False],
                  'objective': ['reg:squarederror'],
                  'learning_rate': [0.005, 0.0075, 0.008, 0.01],
                  'max_depth': [1, 2, 3, 4, 6],
                  'max_features': [0.2, 0.5],
                  'n_estimators': [70, 85, 100, 125, 140, 180],
                  'colsample_bylevel': [0.2, 0.5],
                  'subsample': [0.85, None]}

    tscv = TimeSeriesSplit(n_splits=n_split, max_train_size=10752, test_size=672)
    xgb_gs = GridSearchCV(xgboost.XGBRegressor(), param_grid=parameters, cv=tscv,
                          scoring='neg_mean_absolute_percentage_error')
    xgb_gs.fit(x, y)
    xgb_gs.transform(model)


def anomaly_simulator(dataset):
    total_error = []
    filename = 'Datasets/' + dataset + '/' + i + 'final.csv'
    df = pd.read_csv(filename, index_col='Datetime')
    last_date = datetime.fromisoformat(df.index[-1])
    train_start_date = datetime.fromisoformat(global_variables.first_d[dataset])
    train_end_date = train_start_date + timedelta(weeks=16)
    test_end_date = train_end_date + timedelta(weeks=1)

    features = global_variables.pearson[dataset]
    model = global_variables.best_model[dataset]

    x = df[features]
    y = df['Consumption(Wh)']

    while test_end_date < last_date:
        x_train = x[str(train_start_date):str(train_end_date)]
        y_train = y[str(train_start_date):str(train_end_date)]
        x_test = x[str(train_end_date):str(test_end_date)]
        y_test = y[str(train_end_date):str(test_end_date)]
        model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)
        y_pred = model.predict(x_test)
        aggregated = helper.aggregate(y_test, y_pred)
        MAPE = mean_absolute_percentage_error(aggregated[0], aggregated[1])
        total_error.append(MAPE)
        if MAPE > 0.4:
            print('Anomaly detected : ' + str(MAPE))
            best_model_search(dataset, str(test_end_date), model)
            print(model.get_params)
        else:
            print("MAPE : " + str(MAPE))
        train_start_date = train_start_date + timedelta(weeks=1)
        train_end_date = train_start_date + timedelta(weeks=16)
        test_end_date = train_end_date + timedelta(weeks=1)
    print("Average MAPE : " + str(np.mean(total_error)))


if __name__ == '__main__':
    for i in ['01']:  # , '02', '03', '04', '05', '06', '07', '08']:
        anomaly_simulator(i)
