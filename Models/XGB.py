from datetime import datetime, timedelta

import pandas as pd
import xgboost
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import numpy as np

import global_variables
# import evaluation
import helper


def XGB_regressor_model(set, scale=False, show=False):
    """
    train a random forest model with the dataset 'filename'
    - set (optional) : provide train and test sets
    - scale (boolean) : scale data if true
    """
    x_train = set[0]
    y_train = set[1]
    x_test = set[2]
    y_test = set[3]

    if scale:
        sc = StandardScaler()
        scaler = sc.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    model = xgboost.XGBRegressor(booster='gbtree',
                                 eval_metric='rmse',
                                 early_stopping_rounds=20,
                                 objective='reg:squarederror',
                                 learning_rate=0.008,  # best 0.0075
                                 max_depth=4,  # best 6
                                 n_estimators=135,  # best 125
                                 subsample=0.95,
                                 colsample_bylevel=0.5)

    model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], verbose=False)

    if show:
        y_predict = model.predict(x_test)
        aggregated = helper.aggregate(y_test.values, y_predict)
        helper.plot_model(y_test.values, y_predict, 'XGB')
        helper.plot_model(aggregated[0], aggregated[1], 'XGB - testing - dataset01 - (2021-02-27)')
        print("Accuracy : ")
        print(helper.evaluate_model(y_test.values, y_predict))
        print("Accuracy for aggregated values : ")
        print(helper.evaluate_model(aggregated[0], aggregated[1]))

    return model


if __name__ == '__main__':
    """
    total_error = []
    for i in ['05']:  # , '02', '03', '04', '05', '06', '07', '08']:
        filename = '../Datasets/' + i + '/' + i + 'final.csv'
        df = pd.read_csv(filename, index_col='Datetime')
        features = global_variables.pearson[i]
        # features = global_variables.spearman[i]
        # features = global_variables.mutual_i[i]
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
            x_test = x[str(train_last_date + timedelta(days=3)):str(train_last_date + timedelta(days=10))]
            y_test = y[str(train_last_date + timedelta(days=3)):str(train_last_date + timedelta(days=10))]
            trained_model = XGB_regressor_model(set=[x_train, y_train, x_test, y_test])
            y_predict = trained_model.predict(x_test)
            aggregated = helper.aggregate(y_test.values, y_predict)
            MAPE = round(mean_absolute_percentage_error(aggregated[0], aggregated[1]), 6)
            errors.append(MAPE)
            total_error.append((MAPE))

            train_first_date = train_first_date + timedelta(weeks=3)
        print("Average MAPE for model " + 'XGB' + " with feature selection strategy " + 'Spearman'
              + " and dataset " + i)
        print(np.mean(errors))
    print("total error :" + str(np.mean(total_error)))

    """
    for i in ['01']:  # , '02', '03', '04', '05', '06', '07', '08']:  #
        filename = '../Datasets/' + i + '/' + i + 'final.csv'
        features = ['shortwave_radiation', 'direct_normal_irradiance', 'dewpoint_2m',
                    'Prev_4w_mean_cons', 'Prev_4d_mean_cons']
        df = pd.read_csv(filename, index_col='Datetime')

        train_set = df['2020-11-24 00:00:00':'2021-02-24 00:00:00']
        test_set = df['2021-02-27 00:00:00':'2021-02-28 00:00:00']
        train_visu = df['2021-02-21 00:00:00':'2021-02-24 00:00:00']

        x_train = np.transpose([train_set[var].to_numpy() for var in features])
        y_train = train_set["Consumption(Wh)"]
        x_test = np.transpose([test_set[var].to_numpy() for var in features])
        y_test = test_set["Consumption(Wh)"]
        xgb = XGB_regressor_model(set=[x_train, y_train, x_test, y_test], show=True)

        print('MAPE:' + str(np.round(mean_absolute_percentage_error(y_train.values, xgb.predict(x_train)))))
        x_train_visu = train_visu[features]
        y_train_visu = train_visu['Consumption(Wh)']

        plt.plot(y_train_visu, label='Training data')
        plt.plot(xgb.predict(x_train_visu), label='fitted model')
        plt.title("XGB - training - dataset01 - (2021-02-21, 2021-02-24)")
        plt.xticks([''])
        plt.legend()
        plt.ylabel("Consumption(Wh)")
        plt.show()
