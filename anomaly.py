from datetime import datetime, timedelta
import numpy as np
from helper import *
from helperv2 import *
from Models.RandomForest import random_forest_model
from Models.XGB import XGB_regressor_model
import pandas as pd

from Models.mlp_regression import mlp_model

features = ["Minutes", "Day", "Week", "Weekend", "Month", "Temperature",
            "Humidity", "Pressure", "Wind_speed", "Wind_direction", "Snowfall",
            "Snow_depth", "Irradiation", "Rainfall", 'Previous_4d_mean_cons']

models = {"XGB": XGB_regressor_model, 'R_F': random_forest_model, "MLP": mlp_model,
                 "POLY": polynomial_regressor, "KNN": knn_regressor, "SVM": SVM_regressor_model}


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

    if results['MAE'] > 50:
        print("Anomaly detected : " + str(results))
        return True
    else:
        print(results)


def anomaly_simulator(df, train_n_weeks, n_week, n_days, dataset):

    dates = [datetime.fromisoformat(d) for d in df.index]
    test_start_date = dates[-1]-timedelta(weeks=n_week)
    train_start_date = test_start_date-timedelta(weeks=train_n_weeks)

    find_models_features(df[:str(test_start_date)], features.copy(), dataset, train_n_weeks)
    best_model = select_best_model(dataset)
    #best_model = "XGB"
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
        plot_model(aggregated[0], aggregated[1], best_model)
        anomaly = anomaly_detection(aggregated[0], aggregated[1])
        if anomaly:
            anomaly_cnt += 1
            if anomaly_cnt == 3:
                find_models_features(df[:str(test_start_date)], features.copy(), dataset, train_n_weeks)
                best_model = select_best_model(dataset)
                variables = get_feature(best_model, dataset)
                print('\n' + best_model + '\n' + str(features) + '\n')
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


if __name__ == '__main__':
    df = pd.read_csv('Datasets/09/09.csv', index_col='Datetime')
    anomaly_simulator(df['2020-06-09 00:00:00':], 12, 2, 1, '09')

    #df = pd.read_csv('Datasets/09.csv', index_col='Datetime')
    #anomaly_simulator(df['2020-06-09 00:00:00':], 8, 3, '09_test')