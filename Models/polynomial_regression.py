import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
import helper



def final_model(filename, variables):
    df = pd.read_csv(filename)
    # x = np.transpose([df[var].to_numpy() for var in variables])
    x = df[["Minutes", "Day", "Weekend", "Week", "Month"]]
    y = df["Consumption(Wh)"]
    size = len(df) - 96
    x_train = x[:size]
    y_train = y[:size]
    x_test = x[size:]
    y_test = y[size:]
    polynomial_features = PolynomialFeatures(degree=4)
    poly_x = polynomial_features.fit_transform(x_train)
    model = LinearRegression()
    model.fit(poly_x, y_train)
    #x_test = np.transpose([test_df[var].to_numpy() for var in variables])
    x_test_poly = polynomial_features.fit_transform(x_test)
    #y_test = test_df["Consumption(Wh)"]
    y_predict = model.predict(x_test_poly)

    helper.evaluate_model(y_test.values, y_predict)
    helper.plot_model(y_test.values, y_predict)

def polynomial_regressor(set, scale=False, show=False):
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

    polynomial_features = PolynomialFeatures(degree=4)
    poly_x = polynomial_features.fit_transform(x_train)
    model = LinearRegression()
    model.fit(poly_x, y_train)

    if show:
        y_predict = model.predict(x_test)
        aggregated = helper.aggregate(y_test.values, y_predict)
        helper.plot_model(y_test.values, y_predict, 'Polynomial')
        helper.plot_model(aggregated[0], aggregated[1], 'Polynomial_aggregated')
        print("Accuracy : ")
        print(helper.evaluate_model(y_test.values, y_predict))
        print("Accuracy for aggregated values : ")
        print(helper.evaluate_model(aggregated[0], aggregated[1]))

    return model



if __name__ == '__main__':
    variables = ["Day", "Week", "Weekend", "Month", "Temperature", "Humidity", "Pressure",
                 "Wind speed", "Wind direction", "Snowfall", "Snow depth", "Irradiation", "Rainfall", "Minutes"]

    best10 = ['Minutes', 'Weekend', 'Temperature', 'Irradiation', 'Month', 'Wind direction', 'Wind speed', 'Pressure', 'Day', 'Week', 'Humidity']
    best09 = ['Minutes', 'Week', 'Temperature', 'Irradiation', 'Pressure', 'Snow depth', 'Month', 'Wind direction', 'Weekend', 'Day', 'Humidity', 'Wind speed']

    final_model('../Datasets/10/one_year_10.csv', helper.get_features('one_year_10.csv'))
    final_model('../Datasets/09/one_year_09.csv', helper.get_features('one_year_09.csv'))
