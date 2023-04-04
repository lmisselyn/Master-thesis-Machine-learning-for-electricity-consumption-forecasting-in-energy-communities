from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_regression
import helper
from Models import RandomForest, XGB, SVM


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
    fig1.savefig('../plots/' + model_name)
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
        trained_model = model(set=[x_train, y_train, x_test, y_test])
        y_predict = trained_model.predict(x_test)
        aggregated = aggregate(y_test.values, y_predict)
        acc = evaluate_model(aggregated[0], aggregated[1])

        for k in acc.keys():
            results[k].append(acc[k])
    for k in results.keys():
        results[k] = np.mean(results[k])
    print(results)
    return results


def one_week_test2(filename, model, variables):
    """
    Take a filename, a model and some variables
    Return the mean of accuracy measures over the last week of the dataset
    """
    df = pd.read_csv(filename, index_col='Datetime')
    x = df[variables]
    y = df["Consumption(Wh)"]


    dates = [datetime.fromisoformat(d) for d in df.index]
    results = {"MAE": [], "MSE": [], "RMSE": [], "MAPE": []}

    first_date = dates[0] + timedelta(weeks=1)
    last_date = dates[-1]+timedelta(minutes=30)
    train_last_date = last_date-timedelta(weeks=1)
    print(train_last_date)
    x_train = x[str(first_date):str(train_last_date)]
    y_train = y[str(first_date):str(train_last_date)]
    xtest = x[str(train_last_date):]
    ytest = y[str(train_last_date):]
    trained_model = model(set=[x_train, y_train, xtest, ytest])

    test_last_date = dates[-1]
    test_first_date = test_last_date+timedelta(minutes=15)
    for i in range(1, 8):
        test_first_date = test_first_date-timedelta(days=1)
        x_test = x[str(test_first_date):str(test_last_date)]
        y_test = y[str(test_first_date):str(test_last_date)]
        test_last_date = test_first_date

        y_predict = trained_model.predict(x_test)
        acc = evaluate_model(y_test, y_predict)

        for k in acc.keys():
            results[k].append(acc[k])
    for k in results.keys():
        results[k] = np.mean(results[k])
    print(results)
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
        '''
        if len(current) == 1 and measures['MAPE'] > 0.6:
            variables.remove(v)
            continue
        '''
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
    variables = ["Minutes", "Day", "Week", "Weekend", "Month", "Temperature", "Humidity", "Pressure",
             "Wind speed", "Wind direction", "Snowfall", "Snow depth", "Irradiation", "Rainfall", 'Previous_4d_mean_cons']

    select_best_features('Datasets/10_test.csv', XGB.XGB_regressor_model, variables)