from datetime import datetime
from datetime import timedelta

import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd

import global_variables
import helper
from Models.mlp_regression import mlp_model
from Models.linear_regression import linear_regression
from Models.KNN import knn_regressor
from Models.SVM import SVM_regressor_model
from Models.RandomForest import random_forest_model
from Models.XGB import XGB_regressor_model
from helper import aggregate
from global_variables import *
var = ['Day', 'Minutes',
       'Weekend', 'temperature_2m', 'relativehumidity_2m',
       'dewpoint_2m', 'apparent_temperature',
       'shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
       'direct_normal_irradiance', 'windspeed_10m',
       'Prev_4d_mean_cons', 'Prev_4w_mean_cons']



if __name__ == '__main__':

    cnt = {}
    for method in [spearman, pearson, mutual_i]:
        for i in ['01', '02', '03', '04', '05', '06', '07', '08']:
            for f in method[i]:
                if f not in cnt:
                    cnt[f] = 1
                else:
                    cnt[f] = cnt[f]+1
    for model in wrapp_r2.keys():
        for i in ['01', '02', '03', '04', '05', '06', '07', '08']:
            for f in wrapp_r2[model][i]:
                if f not in cnt:
                    cnt[f] = 1
                else:
                    cnt[f] = cnt[f]+1

    for model in wrapp_mape.keys():
        for i in ['01', '02', '03', '04', '05', '06', '07', '08']:
            for f in wrapp_mape[model][i]:
                if f not in cnt:
                    cnt[f] = 1
                else:
                    cnt[f] = cnt[f] + 1

    for f in cnt.keys():
        cnt[f] = (cnt[f]/72)*100
    print(cnt)

    """
    model = XGB_regressor_model
    model_name = 'XGB'
    feature_selection_strategy = 'most present'

    total_error = []
    for i in ['01', '02', '03', '04', '05', '06', '07', '08']:
        filename = 'Datasets/' + i + '/' + i + 'final.csv'
        df = pd.read_csv(filename, index_col='Datetime')
        features = most_present_features
        #features = global_variables.pearson[i]
        #features = global_variables.spearman[i]
        errors = []
        last_date = datetime.fromisoformat(df.index[-1])
        train_first_date = datetime.fromisoformat(global_variables.first_d[i])

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
            x_test = x[str(train_last_date + timedelta(days=0)):str(train_last_date + timedelta(days=7))]
            y_test = y[str(train_last_date + timedelta(days=0)):str(train_last_date + timedelta(days=7))]
            trained_model = model(set=[x_train, y_train, x_test, y_test])
            y_predict = trained_model.predict(x_test)
            aggregated = aggregate(y_test.values, y_predict)
            MAPE = round(mean_absolute_percentage_error(aggregated[0], aggregated[1]), 6)
            #print('MAPE : '+str(MAPE))
            #helper.plot_model(aggregated[0], aggregated[1], 'LR - testing - '+str(train_last_date)[:10] + str(MAPE))
            #MAPE = round(mean_absolute_percentage_error(aggregated[0], aggregated[1]), 6)
            errors.append(MAPE)
            total_error.append((MAPE))

            train_first_date = train_first_date + timedelta(weeks=3)
        print("Average MAPE for model " + model_name + " with feature selection strategy " + feature_selection_strategy
              + " and dataset " + i)
        print(np.mean(errors))
    print("total error :" + str(np.mean(total_error)))
    """