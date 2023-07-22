import datetime
from datetime import timedelta
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import pandas as pd
import helper

first_d = {'01': '2020-02-25 00:00:00', '02': '2020-02-15 00:00:00', '03': '2020-02-27 00:00:00',
           '04': '2020-07-24 00:00:00',
           '05': '2020-08-22 00:00:00', '06': '2020-08-25 00:00:00', '07': '2020-08-25 00:00:00',
           '08': '2020-10-06 00:00:00'}
def SVM_regressor_model(set, scale=False, show=False):
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

    model = svm.SVR()
    model.fit(x_train, y_train)

    if show:
        y_predict = model.predict(x_test)
        aggregated = helper.aggregate(y_test.values, y_predict)
        helper.plot_model(y_test.values, y_predict, 'SVM')
        helper.plot_model(aggregated[0], aggregated[1], 'SVM_aggregated')
        print("Accuracy : ")
        print(helper.evaluate_model(y_test.values, y_predict))
        print("Accuracy for aggregated values :")
        print(helper.evaluate_model(aggregated[0], aggregated[1]))
    return model

if __name__ == '__main__':
    features = ['Day', 'Minutes',
                'Weekend', 'temperature_2m', 'relativehumidity_2m',
                'dewpoint_2m', 'apparent_temperature',
                'shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
                'direct_normal_irradiance', 'windspeed_10m',
                'Prev_4d_mean_cons', 'Prev_4w_mean_cons']
    for i in ['01']:#, '02', '03', '04', '05', '06', '07', '08']:
        filename = '../Datasets/' + i + '/' + i + 'final.csv'
        df = pd.read_csv(filename, index_col='Datetime')
        x = df['Day']
        y = df['Consumption(Wh)']
        n_days=1
        train_first_date = datetime.datetime.fromisoformat(first_d[i])
        train_last_date = train_first_date+timedelta(weeks=16)
        test_start_date = train_last_date+timedelta(days=3)
        x_train = x[str(train_first_date):str(test_start_date)]
        y_train = y[str(train_first_date):str(test_start_date)]
        x_test = x[str(test_start_date):str(test_start_date + timedelta(days=n_days))]
        y_test = y[str(test_start_date):str(test_start_date + timedelta(days=n_days))]
        SVM_regressor_model(set=[x_train, y_train, x_test, y_test], show=True)