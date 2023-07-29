import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn import metrics
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
et si les valeurs critiques à des intervalles de confiance de 1%, 5%, 02%
 sont aussi proches que possible des statistiques de l’ADF (Augmented Dickey-Fuller)
 
 https://www.kaggle.com/code/chandra03/how-to-use-sarimax-and-arimax/notebook
 
 Sationarity test :
 https://moncoachdata.com/blog/modele-arima-avec-python/
'''


def get_stationarity(filename):
    timeseries = pd.read_csv(filename, index_col=["Datetime"], usecols=['Consumption(Wh)', 'Datetime'],
                             parse_dates=["Datetime"])
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

def evaluate_model(y, y_pred, show=False):
    MAE = metrics.mean_absolute_error(y, y_pred)
    MSE = metrics.mean_squared_error(y, y_pred)
    RMSE = metrics.mean_squared_error(y, y_pred, squared=False)
    MAPE = round(metrics.mean_absolute_percentage_error(y, y_pred),6)
    if show:
        print("Mean absolute error : " + str(MAE))
        print("Mean squared error : " + str(MSE))
        print("Root Mean square error : " + str(RMSE))
        print("Mean absolute percentage error : " + str(MAPE))
    return {"MAE": MAE, "MSE": MSE, "RMSE": RMSE, "MAPE": MAPE}
def aggregate(y, y_predict):
    aggregated_y = []
    aggregated_y_pred = []
    index = 0
    while index <= len(y) - 4:
        aggregated_y.append(sum(y[index:index + 4]) / 4)
        aggregated_y_pred.append(sum(y_predict[index:index + 4]) / 4)
        index += 4
    return [aggregated_y, aggregated_y_pred]

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


def arima_model(filename=None, set=[], scale=False):

    df = pd.read_csv(filename, index_col='Datetime')
    exo = df[['apparent_temperature', 'diffuse_radiation', 'dewpoint_2m']]
    train_exo = exo['2020-02-25 00:00:00':'2020-05-25 00:00:00']
    test_exo = exo['2020-05-25 00:00:00':'2020-05-30 00:00:00']
    cons = df['Consumption(Wh)']
    train_set = cons['2020-02-25 00:00:00':'2020-05-25 00:00:00']
    test_set = cons['2020-05-25 00:00:00':'2020-05-30 00:00:00']

    model = pm.arima.auto_arima(y=train_set, X=train_exo, seasonal=True, m=96)
    model.summary()

    prediction = model.predict(480, test_exo)
    aggregated = aggregate(test_set["Consumption(Wh)"].values, prediction)
    arima_plot(train_set, test_set, prediction)
    result = evaluate_model(aggregated[0], aggregated[1], show=True)
    print(result)
    return result

def arima_plot(train_set, test_set, prediction):
    # Plot

    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(train_set, label='training')
    plt.plot(test_set, label='actual')
    plt.plot(prediction, label='forecast')

    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
    plt.savefig("arima")
    plt.show()


if __name__ == '__main__':
    # auto_correlation_function('../Datasets/01/01final.csv')
    # get_stationarity('../Datasets/01/01final.csv')
    # adv_test('../Datasets/01/01final.csv')
    print(arima_model("../Datasets/01/01final.csv"))
