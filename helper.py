from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_regression


def plot_model(y, y_predict, model_name):
    fig, ax = plt.subplots()
    ax.plot(y, label='True values')
    ax.plot(y_predict, label='Predicted values')
    ax.set_ylabel("Consumption(Wh)")
    ax.legend(facecolor='white')
    fig1 = plt.gcf()
    #fig1.savefig('../plots/' + model_name)
    plt.title(model_name)
    plt.show()


def evaluate_model(y, y_pred, show=False):
    MAE = metrics.mean_absolute_error(y, y_pred)
    MSE = metrics.mean_squared_error(y, y_pred)
    RMSE = metrics.mean_squared_error(y, y_pred, squared=False)
    MAPE = metrics.mean_absolute_percentage_error(y, y_pred)
    if show:
        print("Mean absolute error : " + str(MAE))
        print("Mean squared error : " + str(MSE))
        print("Root Mean square error : " + str(RMSE))
        print("Mean absolute percentage error : " + str(MAPE))
    return {"MAE": MAE, "MSE": MSE, "RMSE": RMSE, "MAPE": MAPE}


def select_k_best(filename):
    df = pd.read_csv(filename)
    x = df.drop(columns=["Date", "Hour", "Index(Wh)", "Consumption(Wh)"])
    y = df["Consumption(Wh)"]
    selection = SelectKBest(f_regression, k=11)
    selection.fit(x, y)
    features = ["Minutes", "Day", "Weekend", "Week", "Month", "Day of year", "Temperature", "Humidity", "Pressure",
                "Wind speed",
                "Wind direction", "Rainfall", "Snowfall", "Snow depth", "Irradiation"]
    print(selection.get_feature_names_out())


def get_features(filename, model):
    if model == "Random_forest":
        if filename == "Datasets/one_year_09.csv":
            return ['Minutes', 'Snow depth', 'Day', 'Weekend', 'Snowfall']
        elif filename == "Datasets/one_year_10.csv":
            return ['Month', 'Minutes']
    elif model == "MLP":
        if filename == "Datasets/one_year_09.csv":
            return ['Minutes', 'Humidity', 'Snowfall', 'Wind speed']
        elif filename == "Datasets/one_year_10.csv":
            return ['Minutes', 'Snow depth', 'Day', 'Weekend', 'Snowfall']


def aggregate(y, y_predict):
    aggregated_y = []
    aggregated_y_pred = []
    index = 0
    while index <= len(y) - 4:
        aggregated_y.append(sum(y[index:index + 4]) / 4)
        aggregated_y_pred.append(sum(y_predict[index:index + 4]) / 4)
        index += 4
    return [aggregated_y, aggregated_y_pred]




if __name__ == '__main__':
    features = ["Minutes", "Day", "Week", "Weekend", "Month", "Temperature",
                "Humidity", "Pressure", "Wind speed", "Wind direction", "Snowfall",
                "Snow depth", "Irradiation", "Rainfall", 'Previous_4d_mean_cons']


    select = []
    acc = []
    select_best_features('Datasets/10.csv', XGB_regressor_model, features, select, acc)
    print(select)
    print(acc)