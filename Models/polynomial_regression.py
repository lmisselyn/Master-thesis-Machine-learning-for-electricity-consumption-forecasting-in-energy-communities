import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import helper


def regress_visu(filename, variables):
    df = pd.read_csv(filename)
    for v in variables:
        x = df[v]
        y = df["Consumption(Wh)"]
        model = np.poly1d(np.polyfit(x, y, 3))
        fig, ax = plt.subplots()
        ax.plot(x, y, linewidth=0, marker='s', label='Data points')
        ax.plot(x, model(x), label='polynomial_regress')
        ax.set_xlabel(v)
        ax.set_ylabel("Consumption(Wh)")
        ax.legend(facecolor='white')
        plt.show()


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

def variable_selection(filename, variables):
    train_df, validation_df, test_df = helper.make_sets(filename)
    selected_var = []
    best_accuracy = -10000
    y = train_df["Consumption(Wh)"]
    iter = 0

    while len(variables) > 0:
        max_accur = -10000
        best_var = ""

        for v in variables:
            y_validation = validation_df["Consumption(Wh)"]
            model = LinearRegression()
            x = []
            x_validation = []

            # first selection
            if len(selected_var) == 0:
                x = train_df[v].to_numpy().reshape(-1, 1)
                x_validation = validation_df[v].to_numpy().reshape(-1, 1)
            else:
                x = [train_df[var].to_numpy() for var in selected_var]
                x.append(train_df[v].to_numpy())
                x_validation = [validation_df[var].to_numpy() for var in selected_var]
                x_validation.append(validation_df[v].to_numpy())
                x = np.array(x).transpose()
                x_validation = np.array(x_validation).transpose()

            polynomial_features = PolynomialFeatures(degree=10)
            x_poly = polynomial_features.fit_transform(x)
            model.fit(x_poly, y)
            x_valid_poly = polynomial_features.fit_transform(x_validation)
            y_predict = model.predict(x_valid_poly)
            cvrmse = (np.sqrt(mean_squared_error(y_validation, y_predict))) / y_validation.mean()
            MBE = np.mean(y_predict - y_validation)
            MAE = np.mean(np.abs(y_predict - y_validation))
            #accuracy = 0.6 * (1 - cvrmse) + 0.4 * (1 - MBE)
            accuracy = -MAE
            if accuracy > max_accur:
                max_accur = accuracy
                best_var = v

        # return if no variable selected
        if best_var != "":
            selected_var.append(best_var)
            variables.remove(best_var)
        else:
            return selected_var, best_accuracy
        if max_accur > best_accuracy:
            # return if accuracy gain is low
            if best_accuracy != -10000 and max_accur - best_accuracy < 0.01:
                best_accuracy = max_accur
                return selected_var, best_accuracy
            best_accuracy = max_accur
    return selected_var, best_accuracy




if __name__ == '__main__':
    variables = ["Day", "Week", "Weekend", "Month", "Temperature", "Humidity", "Pressure",
                 "Wind speed", "Wind direction", "Snowfall", "Snow depth", "Irradiation", "Rainfall", "Minutes"]

    best10 = ['Minutes', 'Weekend', 'Temperature', 'Irradiation', 'Month', 'Wind direction', 'Wind speed', 'Pressure', 'Day', 'Week', 'Humidity']
    best09 = ['Minutes', 'Week', 'Temperature', 'Irradiation', 'Pressure', 'Snow depth', 'Month', 'Wind direction', 'Weekend', 'Day', 'Humidity', 'Wind speed']

    final_model('../Datasets/one_year_10.csv', helper.get_features('one_year_10.csv'))
    final_model('../Datasets/one_year_09.csv', helper.get_features('one_year_09.csv'))
