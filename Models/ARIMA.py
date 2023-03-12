import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from statsmodels.graphics.tsaplots import plot_acf

'''
Statistiques roulantes : Tracer la moyenne mobile et l’écart-type mobile. 
La série temporelle est stationnaire si elle reste constante dans le temps 
(à l’œil nu, regardez si les lignes sont droites et parallèles à l’axe des x)
Test de Dickey-Fuller augmenté (ADF) : La série temporelle est considérée 
comme stationnaire si la valeur p est faible (selon l’hypothèse nulle) 
et si les valeurs critiques à des intervalles de confiance de 1%, 5%, 10%
 sont aussi proches que possible des statistiques de l’ADF (Augmented Dickey-Fuller)
 
 https://www.kaggle.com/code/chandra03/how-to-use-sarimax-and-arimax/notebook
'''

def get_stationarity(filename):

    df = pd.read_csv(filename)
    timeseries = df[['Minutes', 'Consumption(Wh)']]
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
    df = pd.read_csv(filename, usecols=['Consumption(Wh)'])
    plot_acf(df)
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv('../Datasets/one_year_10.csv')
    model = ARIMA(df.values, order=(1, 1, 2))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())