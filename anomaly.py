from datetime import datetime, timedelta
import numpy as np
from helper import *
from helperv2 import *
from Models.linear_regression import linear_regression
from Models.RandomForest import random_forest_model
from Models.XGB import XGB_regressor_model
from Models.mlp_regression import mlp_model
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error


features = ['Day', 'Minutes',
           'Weekend', 'temperature_2m', 'relativehumidity_2m',
           'dewpoint_2m', 'apparent_temperature',
           'shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
           'direct_normal_irradiance', 'windspeed_10m',
           'Prev_4d_mean_cons', 'Prev_4w_mean_cons']

first_d = {'01': '2020-02-25 00:00:00', '02': '2020-02-15 00:00:00', '03': '2020-02-27 00:00:00',
           '04': '2020-07-24 00:00:00',
           '05': '2020-08-22 00:00:00', '06': '2020-08-25 00:00:00', '07': '2020-08-25 00:00:00',
           '08': '2020-10-06 00:00:00'}

models = {"LR": linear_regression, "XGB": XGB_regressor_model, 'R_F': random_forest_model, "MLP": mlp_model,
          "KNN": knn_regressor, "SVM": SVM_regressor_model}


def anomaly_detection(y_true, y_predict):
    results = {"MAE": [], "MSE": [], "RMSE": [], "MAPE": []}
    indice=0
    while indice < len(y_true):
        acc = evaluate_model(
            y_true[indice:indice+24], y_predict[indice:indice+24])
        for k in acc.keys():
            results[k].append(acc[k])
        indice += 24
    for k in results.keys():
        results[k] = np.mean(results[k])

    if results['MAPE'] > 0.5:
        print("Anomaly detected : " + str(results))
        return True
    else:
        print(results)


def anomaly_simulator(df, train_n_weeks, n_week, n_days, dataset):

    total_error = []
    dates = [datetime.fromisoformat(d) for d in df.index]
    test_start_date = dates[-1]-timedelta(weeks=n_week)
    train_start_date = test_start_date-timedelta(weeks=train_n_weeks)

    find_models_features(df[:str(test_start_date)], features.copy(), dataset, train_n_weeks)
    best_model = select_best_model(dataset)
    variables = get_feature(best_model, dataset)

    print('\n' + best_model + '\n' + str(variables) + '\n')

    x = df[variables]
    y = df['Consumption(Wh)']
    x_train = x[str(train_start_date):str(test_start_date)]
    y_train = y[str(train_start_date):str(test_start_date)]
    x_test = x[str(test_start_date):str(test_start_date+timedelta(days=n_days))]
    y_test = y[str(test_start_date):str(test_start_date + timedelta(days=n_days))]

    model = models[best_model]
    trained_model = model(set=[x_train, y_train, x_test, y_test])

    anomaly_cnt = 0

    while test_start_date < dates[-1]:
        end_date = test_start_date+timedelta(days=n_days)
        x_test = x[str(test_start_date):str(end_date)]
        y_test = y[str(test_start_date):str(end_date)]
        y_predict = trained_model.predict(x_test)
        aggregated = aggregate(y_test.values, y_predict)
        MAPE = round(mean_absolute_percentage_error(aggregated[0], aggregated[1]), 6)
        total_error.append(MAPE)
        #plot_model(aggregated[0], aggregated[1], best_model)
        anomaly = anomaly_detection(aggregated[0], aggregated[1])
        if anomaly:
            anomaly_cnt += 1
            if anomaly_cnt == 5:
                find_models_features(df[:str(test_start_date)], features.copy(), dataset, train_n_weeks)
                best_model = select_best_model(dataset)
                variables = get_feature(best_model, dataset)
                print('\n' + best_model + '\n' + str(variables) + '\n')
                x = df[variables]
            x_train = x[str(train_start_date):str(test_start_date)]
            y_train = y[str(train_start_date):str(test_start_date)]
            x_test = x[str(test_start_date):str(end_date)]
            y_test = y[str(test_start_date):str(end_date)]
            trained_model = model(set=[x_train, y_train, x_test, y_test])
        else:
            anomaly_cnt = 0
        test_start_date = end_date
        train_start_date = test_start_date-timedelta(weeks=train_n_weeks)
    print("Final error :" + str(np.mean(total_error)))


if __name__ == '__main__':
    for i in ['01'] :#, '02', '03', '04', '05', '06', '07', '08']:
        filename = 'Datasets/' + i + '/' + i + 'final.csv'
        df = pd.read_csv(filename, index_col='Datetime')
        print(i)
        anomaly_simulator(df[first_d[i]:], 16, 10, 1, i)
