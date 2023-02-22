import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import helper
from sklearn import metrics


def mlp_model(filename, variables):
    df = pd.read_csv(filename)
    x = np.transpose([df[var].to_numpy() for var in variables])
    y = df["Consumption(Wh)"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    sc = StandardScaler()
    scaler = sc.fit(x_train)
    trainX_scaled = scaler.transform(x_train)
    testX_scaled = scaler.transform(x_test)

    model = MLPRegressor(hidden_layer_sizes=(150, 100, 50),
                           max_iter=300, activation='relu',
                           solver='adam')
    model.fit(trainX_scaled, y_train)

    y_predict = model.predict(testX_scaled)

    helper.evaluate_model(y_test.values, y_predict)
    helper.plot_model(y_test.values, y_predict)

if __name__ == '__main__':
    #best10 = ['Minutes', 'Weekend', 'Temperature', 'Wind direction', 'Wind speed', 'Day of year', 'Day', 'Snowfall', 'Rainfall']
    mlp_model('one_year_10.csv', helper.get_features('one_year_10.csv'))
    mlp_model('one_year_09.csv', helper.get_features('one_year_09.csv'))