import datetime
from datetime import timedelta
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import pandas as pd

#import evaluation
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

    model = svm.SVR(C=5, epsilon=0.4, gamma='scale')
    model.fit(x_train, y_train)

    if show:
        y_predict = model.predict(x_test)
        aggregated = helper.aggregate(y_test.values, y_predict)
        helper.plot_model(y_test.values, y_predict, 'SVM')
        helper.plot_model(aggregated[0], aggregated[1], 'SVM regression - dataset01 - (2021-02-27) ')
        print("Accuracy : ")
        print(helper.evaluate_model(y_test.values, y_predict))
        print("Accuracy for aggregated values :")
        print(helper.evaluate_model(aggregated[0], aggregated[1]))
    return model

if __name__ == '__main__':

    var = ['Day', 'Minutes',
                'Weekend', 'temperature_2m', 'relativehumidity_2m',
                'dewpoint_2m', 'apparent_temperature',
                'shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
                'direct_normal_irradiance', 'windspeed_10m',
                'Prev_4d_mean_cons', 'Prev_4w_mean_cons']


    for i in ['01']: #, '02', '03', '04', '05', '06', '07', '08']:  #
        filename = '../Datasets/' + i + '/' + i + 'final.csv'
        features =  ['apparent_temperature', 'diffuse_radiation', 'dewpoint_2m',
   'Prev_4w_mean_cons', 'Prev_4d_mean_cons']
        df = pd.read_csv(filename, index_col='Datetime')

        train_set = df['2020-11-24 00:00:00':'2021-02-24 00:00:00']
        test_set = df['2021-02-27 00:00:00':'2021-02-28 00:00:00']
        train_visu = df['2021-02-21 00:00:00':'2021-02-24 00:00:00']

        x_train = np.transpose([train_set[var].to_numpy() for var in features])
        y_train = train_set["Consumption(Wh)"]
        x_test = np.transpose([test_set[var].to_numpy() for var in features])
        y_test = test_set["Consumption(Wh)"]
        svm = SVM_regressor_model(set=[x_train, y_train, x_test, y_test], show=True)
        print(features)

        x_train_visu = train_visu[features]
        y_train_visu = train_visu['Consumption(Wh)']

        plt.plot(y_train_visu,  label='Training data')
        plt.plot(svm.predict(x_train_visu),  label='fitted model')
        plt.title("SVM visualisation - dataset01 - (2021-02-21, 2021-02-24)")
        plt.xticks([''])
        plt.legend()
        plt.ylabel("Consumption(Wh)")
        plt.show()