from datetime import datetime, timedelta
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from helper import *
import warnings

def mlp_model(set, scale=False, show=False):
    warnings.filterwarnings('ignore')
    """
    train a random multi layer perceptron model with the dataset 'filename'
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
        x_train = pd.DataFrame(scaler.transform(x_train), x_train.index, x_train.columns)
        x_test = pd.DataFrame(scaler.transform(x_test), x_test.index, x_train.columns)

    model = MLPRegressor(
        hidden_layer_sizes=(10, 10, 10), #best (10, 10, 10)
        activation='relu', #best relu
        solver='adam',
        learning_rate='constant',
        learning_rate_init=0.0001,
        max_iter=1500,
        n_iter_no_change=25,
        early_stopping=True,
        validation_fraction=0.15)

    model.fit(x_train, y_train)

    if show:
        y_predict = model.predict(x_test)
        aggregated = aggregate(y_test.values, y_predict)
        plot_model(y_test.values, y_predict, 'mlp')
        plot_model(aggregated[0], aggregated[1], 'XGB - test - dataset01 - (2021-02-27)')
        print('Accuracy :')
        print(evaluate_model(y_test.values, y_predict))
        print("Aggregated accuracy :")
        print(evaluate_model(aggregated[0], aggregated[1]))

    return model




if __name__ == '__main__':

    variables = ['Day', 'Minutes',
           'Weekend', 'relativehumidity_2m',
           'dewpoint_2m', 'apparent_temperature',
           'shortwave_radiation', 'direct_radiation',
                 'windspeed_10m',
           'Prev_4d_mean_cons', 'Prev_4w_mean_cons']

    for i in ['01']: #, '02', '03', '04', '05', '06', '07', '08']:  #
        filename = '../Datasets/' + i + '/' + i + 'final.csv'
        #features = ['shortwave_radiation', 'direct_normal_irradiance', 'dewpoint_2m',
         #           'Prev_4w_mean_cons', 'Prev_4d_mean_cons']
        features = variables
        df = pd.read_csv(filename, index_col='Datetime')

        train_set = df['2020-11-24 00:00:00':'2021-02-24 00:00:00']
        test_set = df['2021-02-27 00:00:00':'2021-02-28 00:00:00']
        train_visu = df['2021-02-21 00:00:00':'2021-02-24 00:00:00']

        x_train = np.transpose([train_set[var].to_numpy() for var in features])
        y_train = train_set["Consumption(Wh)"]
        x_test = np.transpose([test_set[var].to_numpy() for var in features])
        y_test = test_set["Consumption(Wh)"]
        mlp = mlp_model(set=[x_train, y_train, x_test, y_test], show=True)

        print('MAPE:' + str(np.round(mean_absolute_percentage_error(y_train.values, mlp.predict(x_train)), 3)))
        x_train_visu = train_visu[features]
        y_train_visu = train_visu['Consumption(Wh)']

        plt.plot(y_train_visu,  label='Training data')
        plt.plot(mlp.predict(x_train_visu),  label='fitted model')
        plt.title("MLP training - dataset01 - (2021-02-21, 2021-02-24)")
        plt.xticks([''])
        plt.legend()
        plt.ylabel("Consumption(Wh)")
        plt.show()
