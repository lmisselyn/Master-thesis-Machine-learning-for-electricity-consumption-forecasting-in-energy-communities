from helper import evaluate_model
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import register_matplotlib_converters


register_matplotlib_converters()
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import pmdarima as pm

'''
Statistiques roulantes : Tracer la moyenne mobile et l’écart-type mobile. 
La série temporelle est stationnaire si elle reste constante dans le temps 
(à l’œil nu, regardez si les lignes sont droites et parallèles à l’axe des x)
Test de Dickey-Fuller augmenté (ADF) : La série temporelle est considérée 
comme stationnaire si la valeur p est faible (selon l’hypothèse nulle) 
et si les valeurs critiques à des intervalles de confiance de 1%, 5%, 10%
 sont aussi proches que possible des statistiques de l’ADF (Augmented Dickey-Fuller)
 
 https://www.kaggle.com/code/chandra03/how-to-use-sarimax-and-arimax/notebook
 
 Sationarity test :
 https://moncoachdata.com/blog/modele-arima-avec-python/
'''

def get_stationarity(filename):

    timeseries = pd.read_csv(filename, index_col=["Datetime"], usecols=['Consumption(Wh)', 'Datetime'], parse_dates=["Datetime"])
    timeseries.plot()
    plt.show()

    # Statistiques mobiles
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()

    # tracé statistiques mobiles
    original = plt.plot(timeseries, color='blue', label='Origine')
    mean = plt.plot(rolling_mean, color='red', label='Moyenne Mobile')
    std = plt.plot(rolling_std, color='black', label='Ecart-type Mobile')
    plt.legend(loc='best')
    plt.title('Moyenne et écart-type Mobiles')
    plt.show(block=False)

    # Test Dickey–Fuller :
    result = adfuller(timeseries['Consumption(Wh)'])
    print('Statistiques ADF : {}'.format(result[0]))
    print('p-value : {}'.format(result[1]))
    print('Valeurs Critiques :')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

def auto_correlation_function(filename):
    timeseries = pd.read_csv(filename, index_col=["Datetime"], usecols=['Consumption(Wh)', 'Datetime'],
                             parse_dates=["Datetime"])
    plot_acf(timeseries)
    plt.show()

def partial_auto_correlation(filename):
    timeseries = pd.read_csv(filename, index_col=["Datetime"], usecols=['Consumption(Wh)', 'Datetime'],
                             parse_dates=["Datetime"])
    plot_pacf(timeseries)
    plt.show()

def adv_test(filename):
    series = pd.read_csv(filename, index_col=["Datetime"], usecols=['Consumption(Wh)', 'Datetime'],
                             parse_dates=["Datetime"])
    adf = pm.arima.ADFTest(alpha=0.05)
    print(adf.should_diff(series))


def arima_model(filename):
    timeseries = pd.read_csv(filename, index_col=["Datetime"], usecols=['Consumption(Wh)', 'Datetime'],
                             parse_dates=["Datetime"])
    train_set = timeseries['2021-01-06 00:00:00':'2021-02-06 23:45:00']
    test_set = timeseries['2021-02-06 23:45:00':]

    model = pm.arima.auto_arima(
            y=train_set,
            start_p=0,
            d=1,
            start_q=0,
            max_p=5,
            max_d=2,
            max_q=5,
            start_P=1,
            D=1,
            start_Q=0,
            max_P=5,
            max_D=5,
            max_Q=5,
            max_order=5,
            m=365,
            seasonal=True,
            stationary=False,
            information_criterion='aic',
            alpha=0.05,
            test='kpss',
            seasonal_test='ocsb',
            stepwise=True,
            n_jobs=1,
            method='lbfgs',
            maxiter=50,
            suppress_warnings=True,
            random_state=None,
            n_fits=10,
            scoring='mse',
            error_action='trace')

    model.summary()
    prediction = model.predict(288)
    #arima_plot(train_set, test_set, prediction)
    return evaluate_model(test_set["Consumption(Wh)"].values, prediction, show=True)


def arima_plot(train_set, test_set, prediction):
    # Plot
    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(train_set, label='training')
    plt.plot(test_set, label='actual')
    plt.plot(prediction, label='forecast')

    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()

if __name__ == '__main__':
    #auto_correlation_function('../test.csv')
    #get_stationarity('../test.csv')
    #adv_test("../test.csv")
    arima_model("test.csv")
