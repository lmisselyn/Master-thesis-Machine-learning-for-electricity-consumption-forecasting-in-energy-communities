from datetime import datetime, timedelta

import numpy as np

import helper
import helperv2
from Models.XGB import XGB_regressor_model
import pandas as pd

features = ["Minutes", "Day", "Week", "Weekend", "Month", "Temperature",
                         "Humidity", "Pressure", "Wind speed", "Wind direction", "Snowfall",
                         "Snow depth", "Irradiation", "Rainfall", 'Previous_4d_mean_cons']


def anomaly_detection(y_true, y_predict):
    results = {"MAE": [], "MSE": [], "RMSE": [], "MAPE": []}

    indice=0
    while indice < len(y_true):
        acc = helperv2.evaluate_model(
            y_true[indice:indice+24], y_predict[indice:indice+24])
        for k in acc.keys():
            results[k].append(acc[k])
        indice += 24
    for k in results.keys():
        results[k] = np.mean(results[k])

    if results['MAPE'] > 0.4:
        print("Anomaly detected : " + str(results))
        return True
    else:
        print(results)


def one_month_anomaly(filename):
    df = pd.read_csv(filename, index_col='Datetime')
    dates = [datetime.fromisoformat(d) for d in df.index]

    begin_date = dates[-1]-timedelta(weeks=4)+timedelta(minutes=15)
    #best_model = helperv2.select_best_model()
    variables = helperv2.select_best_features(df[:str(begin_date)], XGB_regressor_model, features)[0]
    #variables = ['Minutes', 'Month', 'Weekend', 'Temperature', 'Snowfall', 'Pressure']
    x = df[variables]
    y = df['Consumption(Wh)']

    x_train = x[:str(begin_date)]
    y_train = y[:str(begin_date)]
    x_test = x[str(begin_date):str(begin_date+timedelta(days=3))]
    y_test = y[str(begin_date):str(begin_date + timedelta(days=3))]
    model = XGB_regressor_model(set=[x_train, y_train, x_test, y_test])

    while begin_date < dates[-1]:
        end_date = begin_date+timedelta(days=3)
        x_test = x[str(begin_date):str(end_date)]
        y_test = y[str(begin_date):str(end_date)]
        y_predict = model.predict(x_test)
        aggregated = helperv2.aggregate(y_test.values, y_predict)
        helper.plot_model(aggregated[0], aggregated[1], 'XGB')
        anomaly = anomaly_detection(aggregated[0], aggregated[1])
        if anomaly:
            variables = helperv2.select_best_features(df[:str(begin_date)], XGB_regressor_model, features)[0]
            x = df[variables]
            y = df['Consumption(Wh)']
            x_train = x[:str(begin_date)]
            y_train = y[:str(begin_date)]
            x_test = x[str(begin_date):str(end_date)]
            y_test = y[str(begin_date):str(end_date)]
            model = XGB_regressor_model(set=[x_train, y_train, x_test, y_test])

        begin_date = end_date



if __name__ == '__main__':
    one_month_anomaly('Datasets/10_test.csv')