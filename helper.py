import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_regression
from datetime import datetime

import RandomForest


def make_sets(filename):
    """
    parameters ;
        filename : name of the csv file
    return ;
        training set, validation set, test set
    """
    df = pd.read_csv(filename)
    data = [[], [], []]
    i = 0
    for j in range(len(df)):
        data[i].append(df.loc[j])
        i += 1
        if i > 2:
            i = 0
    train_df = pd.DataFrame(data[0], columns=df.columns)
    validation_df = pd.DataFrame(data[1], columns=df.columns)
    test_df = pd.DataFrame(data[2], columns=df.columns)
    return train_df, validation_df, test_df


def plot_model(y, y_predict, model_name):
    fig, ax = plt.subplots()
    ax.plot(y, label='True values')
    ax.plot(y_predict, label='Predicted values')
    ax.set_ylabel("Consumption(Wh)")
    ax.legend(facecolor='white')
    fig1 = plt.gcf()
    fig1.savefig('plots/' + model_name)
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


def get_features(filename):
    if filename == "one_year_10.csv":
        return ['Minutes', 'Day', 'Week', 'Weekend', 'Temperature', 'Humidity',
                'Pressure', 'Wind speed', 'Wind direction', 'Irradiation']
    else:
        return ['Minutes', 'Week', 'Month', 'Day of year', 'Temperature',
                'Humidity', 'Wind speed', 'Snowfall', 'Snow depth', 'Irradiation']


def one_week_test(filename, model, variables):
    """
    Take a filename, a model and some variables
    Return the mean of accuracy measures over the last week of the dataset
    """
    df = pd.read_csv(filename)
    x = np.transpose([df[var].to_numpy() for var in variables])
    y = df["Consumption(Wh)"]

    results = {"MAE": [], "MSE": [], "RMSE": [], "MAPE": []}
    for i in range(1, 8):
        index = len(x) - i * 96
        x_train = x[:index]
        y_train = y[:index]
        x_test = x[index:index + 96]
        y_test = y[index:index + 96]
        acc = model(set=[x_train, y_train, x_test, y_test])

        for k in acc.keys():
            results[k].append(acc[k])
    for k in results.keys():
        results[k] = np.mean(results[k])
    return results


def aggregate(y, y_predict):
    aggregated_y = []
    aggregated_y_pred = []
    index = 0
    while index <= len(y) - 4:
        aggregated_y.append(sum(y[index:index + 4]) / 4)
        aggregated_y_pred.append(sum(y_predict[index:index + 4]) / 4)
        index += 4
    return [aggregated_y, aggregated_y_pred]


def select_best_features(filename, model, variables, selected=[], accuracy=[]):
    min_MAPE = 10000
    best_acc = {}
    best_var = ''

    # If no more variables then return
    if len(variables) == 0:
        return [selected, accuracy]

    for v in variables:
        current = selected.copy()
        current.append(v)
        print("OneWeek : " + str(current) + '\n')
        measures = one_week_test(filename, model, current)
        print("Result for the week : " + str(measures))
        if len(current) == 1 and measures['MAPE'] > 0.6:
            variables.remove(v)
            continue
        if measures['MAPE'] < min_MAPE:
            min_MAPE = measures['MAPE']
            best_var = v
            best_acc = measures

    # If no improvement then return
    if len(accuracy) != 0 and min_MAPE > accuracy[-1]['MAPE']:
        return [selected, accuracy]

    variables.remove(best_var)
    selected.append(best_var)
    accuracy.append(best_acc)
    print("Selected variables : " + str(selected))
    print(accuracy)
    select_best_features(filename, model, variables,
                         selected=selected, accuracy=accuracy)


if __name__ == '__main__':
    res = one_week_test('test.csv', RandomForest.random_forest_model, ['Minutes', 'Day', 'Week', 'Weekend',
                                                                 'Temperature', 'Humidity', 'Irradiation'])
    with open('one_week_10_RF.txt', 'w') as f:
        f.write(res)