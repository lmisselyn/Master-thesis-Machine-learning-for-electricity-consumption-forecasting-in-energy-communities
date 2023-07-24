from datetime import datetime
from datetime import timedelta

import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
from Models.mlp_regression import mlp_model
from Models.linear_regression import linear_regression
from Models.KNN import knn_regressor
from Models.SVM import SVM_regressor_model
from Models.RandomForest import random_forest_model
from Models.XGB import XGB_regressor_model
from helper import aggregate

var = ['Day', 'Minutes',
       'Weekend', 'temperature_2m', 'relativehumidity_2m',
       'dewpoint_2m', 'apparent_temperature',
       'shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
       'direct_normal_irradiance', 'windspeed_10m',
       'Prev_4d_mean_cons', 'Prev_4w_mean_cons']

first_d = {'01': '2020-02-25 00:00:00', '02': '2020-02-15 00:00:00', '03': '2020-02-27 00:00:00',
           '04': '2020-07-24 00:00:00',
           '05': '2020-08-22 00:00:00', '06': '2020-08-25 00:00:00', '07': '2020-08-25 00:00:00',
           '08': '2020-10-06 00:00:00'}

pearson = {'01': ['apparent_temperature', 'diffuse_radiation', 'dewpoint_2m',
                  'Prev_4w_mean_cons', 'Prev_4d_mean_cons'],
           '02': ['relativehumidity_2m', 'Day', 'Weekend', 'Prev_4d_mean_cons',
                  'Prev_4w_mean_cons'], '03': ['shortwave_radiation', 'diffuse_radiation', 'Minutes',
                                               'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
           '04': ['Day', 'Weekend', 'Minutes', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
           '05': ['temperature_2m', 'apparent_temperature', 'dewpoint_2m',
                  'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
           '06': ['apparent_temperature', 'Day', 'windspeed_10m', 'Prev_4d_mean_cons',
                  'Prev_4w_mean_cons'], '07': ['diffuse_radiation', 'apparent_temperature', 'temperature_2m',
                                               'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
           '08': ['apparent_temperature', 'temperature_2m', 'Minutes',
                  'Prev_4d_mean_cons', 'Prev_4w_mean_cons']}

spearman = {'01': ['shortwave_radiation', 'direct_normal_irradiance', 'dewpoint_2m',
                   'Prev_4w_mean_cons', 'Prev_4d_mean_cons'],
            '02': ['Weekend', 'apparent_temperature', 'temperature_2m',
                   'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
            '03': ['apparent_temperature', 'dewpoint_2m', 'Minutes', 'Prev_4d_mean_cons',
                   'Prev_4w_mean_cons'], '04': ['direct_radiation', 'diffuse_radiation', 'shortwave_radiation',
                                                'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
            '05': ['apparent_temperature', 'dewpoint_2m', 'Minutes', 'Prev_4d_mean_cons',
                   'Prev_4w_mean_cons'], '06': ['direct_normal_irradiance', 'Day', 'windspeed_10m', 'Prev_4w_mean_cons',
                                                'Prev_4d_mean_cons'],
            '07': ['direct_normal_irradiance', 'Prev_4d_mean_cons', 'shortwave_radiation',
                   'diffuse_radiation', 'Prev_4w_mean_cons'],
            '08': ['diffuse_radiation', 'relativehumidity_2m', 'Minutes',
                   'Prev_4w_mean_cons', 'Prev_4d_mean_cons']
            }

mutual_i = {'01': ['Minutes', 'dewpoint_2m', 'apparent_temperature', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
            '02': ['temperature_2m', 'dewpoint_2m', 'apparent_temperature', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
            '03': ['Minutes', 'dewpoint_2m', 'shortwave_radiation', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
            '04': ['Minutes', 'temperature_2m', 'apparent_temperature', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
            '05': ['Minutes', 'dewpoint_2m', 'apparent_temperature', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
            '06': ['shortwave_radiation', 'diffuse_radiation', 'windspeed_10m', 'Prev_4d_mean_cons',
                   'Prev_4w_mean_cons'],
            '07': ['Minutes', 'temperature_2m', 'dewpoint_2m', 'apparent_temperature', 'Prev_4w_mean_cons'],
            '08': ['Minutes', 'temperature_2m', 'dewpoint_2m', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons']}

wrapp_r2 = {'LR': {'01': ['Minutes', 'apparent_temperature', 'direct_radiation', 'direct_normal_irradiance',
                          'windspeed_10m', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
                   '02': ['Day', 'Weekend', 'relativehumidity_2m', 'direct_radiation', 'diffuse_radiation',
                          'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
                   '03': ['Day', 'Minutes', 'Weekend', 'relativehumidity_2m', 'diffuse_radiation',
                          'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
                   '04': ['Minutes', 'Weekend', 'shortwave_radiation', 'direct_radiation',
                          'diffuse_radiation', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
                   '05': ['Day', 'Weekend', 'shortwave_radiation', 'diffuse_radiation',
                          'direct_normal_irradiance', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
                   '06': ['Weekend', 'relativehumidity_2m', 'dewpoint_2m', 'diffuse_radiation',
                          'direct_normal_irradiance', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
                   '07': ['dewpoint_2m', 'apparent_temperature', 'diffuse_radiation',
                          'direct_normal_irradiance', 'windspeed_10m', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
                   '08': ['Minutes', 'temperature_2m', 'relativehumidity_2m', 'apparent_temperature',
                          'direct_normal_irradiance', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons']},
            'SVM': {'01': ['Day', 'Weekend', 'apparent_temperature', 'windspeed_10m', 'Prev_4d_mean_cons'],
                    '02': ['relativehumidity_2m', 'dewpoint_2m', 'apparent_temperature', 'Prev_4d_mean_cons',
                           'Prev_4w_mean_cons'],
                    '03': ['Day', 'Weekend', 'dewpoint_2m', 'windspeed_10m', 'Prev_4w_mean_cons'],
                    '04': ['Day', 'Weekend', 'windspeed_10m', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
                    '05': ['Day', 'Weekend', 'dewpoint_2m', 'windspeed_10m', 'Prev_4w_mean_cons'],
                    '06': ['Day', 'relativehumidity_2m', 'shortwave_radiation', 'Prev_4d_mean_cons',
                           'Prev_4w_mean_cons'],
                    '07': ['Day', 'apparent_temperature', 'shortwave_radiation', 'Prev_4d_mean_cons',
                           'Prev_4w_mean_cons'],
                    '08': ['relativehumidity_2m', 'apparent_temperature', 'windspeed_10m', 'Prev_4d_mean_cons',
                           'Prev_4w_mean_cons']},
            'RF': {'01': ['Minutes', 'Weekend', 'dewpoint_2m', 'shortwave_radiation', 'Prev_4d_mean_cons'],
                   '02': ['Minutes', 'dewpoint_2m', 'apparent_temperature', 'Prev_4d_mean_cons',
                          'Prev_4w_mean_cons'],
                   '03': ['Day', 'Minutes', 'shortwave_radiation', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
                   '04': ['Day', 'Minutes', 'shortwave_radiation', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
                   '05': ['Day', 'Minutes', 'Weekend', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
                   '06': ['apparent_temperature', 'shortwave_radiation', 'windspeed_10m', 'Prev_4d_mean_cons',
                          'Prev_4w_mean_cons'],
                   '07': ['Day', 'Weekend', 'relativehumidity_2m', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
                   '08': ['Day', 'Minutes', 'relativehumidity_2m', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons']}}

wrapp_mape = {}

if __name__ == '__main__':

    model = random_forest_model
    model_name = 'RF'
    feature_selection_strategy = 'mutual_info'

    total_error = []
    for i in ['01', '02', '03', '04', '05', '06', '07', '08']:
        filename = 'Datasets/' + i + '/' + i + 'final.csv'
        df = pd.read_csv(filename, index_col='Datetime')
        features = mutual_i[i]
        errors = []
        last_date = datetime.fromisoformat(df.index[-1])
        train_first_date = datetime.fromisoformat(first_d[i])

        x = df[features]
        y = df['Consumption(Wh)']

        for j in range(10):
            train_last_date = train_first_date + timedelta(weeks=16)
            if train_last_date + timedelta(days=6) > last_date:
                print("Date error" + i)
                print(str(train_last_date + timedelta(days=6)))
                print(str(last_date))
                break

            x_train = x[str(train_first_date):str(train_last_date)]
            y_train = y[str(train_first_date):str(train_last_date)]
            x_test = x[str(train_last_date + timedelta(days=3)):str(train_last_date + timedelta(days=10))]
            y_test = y[str(train_last_date + timedelta(days=3)):str(train_last_date + timedelta(days=10))]
            trained_model = model(set=[x_train, y_train, x_test, y_test])
            y_predict = trained_model.predict(x_test)
            aggregated = aggregate(y_test.values, y_predict)
            MAPE = round(mean_absolute_percentage_error(aggregated[0], aggregated[1]), 6)
            errors.append(MAPE)
            total_error.append((MAPE))

            train_first_date = train_first_date + timedelta(weeks=3)
        print("Average MAPE for model " + model_name + " with feature selection strategy " + feature_selection_strategy
              + " and dataset " + i)
        print(np.mean(errors))
    print("total error :" + str(np.mean(total_error)))
