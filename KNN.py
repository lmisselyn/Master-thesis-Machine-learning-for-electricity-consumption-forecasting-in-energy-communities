from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import helper
import numpy as np
import pandas as pd


def random_forest_model(filename):
    df = pd.read_csv(filename)
    #x = np.transpose([df[var].to_numpy() for var in variables])
    x = df[["Minutes", "Day", "Weekend", "Week", "Month"]]
    y = df["Consumption(Wh)"]
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    size = len(df)-96
    x_train = x[:size]
    y_train = y[:size]
    x_test = x[size:]
    y_test = y[size:]

    sc = StandardScaler()
    scaler = sc.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = KNeighborsRegressor(
        n_neighbors=50,
        weights='distance',
        algorithm='auto',
        leaf_size=100, p=2,
        metric='minkowski',
        metric_params=None,
        n_jobs=None)

    model.fit(x_train_scaled, y_train)

    y_predict = model.predict(x_test_scaled)

    helper.evaluate_model(y_test.values, y_predict)
    helper.plot_model(y_test.values, y_predict)


if __name__ == '__main__':

    best10 = ['Minutes', 'Weekend', 'Temperature', 'Wind direction', 'Wind speed', 'Day of year', 'Day', 'Snowfall', 'Rainfall']
    best09 = ['Minutes', 'Week', 'Temperature', 'Irradiation', 'Pressure', 'Snow depth', 'Month', 'Wind direction', 'Weekend', 'Day', 'Humidity', 'Wind speed']
    random_forest_model('one_year_10.csv')
    random_forest_model('one_year_09.csv')