from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn import metrics
from Models import XGB, RandomForest, MTS
from Models.KNN import knn_regressor
from Models.RandomForest import random_forest_model
from Models.SVM import SVM_regressor_model
from Models.XGB import XGB_regressor_model
from Models.mlp_regression import mlp_model



models = {'R_F': random_forest_model, "MLP": mlp_model, "XGB": XGB_regressor_model,
          'KNN': knn_regressor, 'SVM': SVM_regressor_model}

def select_best_model(df, features, dataset):
    """
    Select the best model for the specific dataset
    """
    find_models_features(df, features, dataset)



def find_models_features(df, features, dataset):
    """
    Find the best features for each model and store results in a text file
    """
    for model in models.keys():
        selected = []
        accuracy = []
        select_best_features(df, models[model], features, selected, accuracy)
        with open('Features/'+model+'_'+dataset+'.txt', 'w') as file:
            file.write(str(selected)+'\n'+str(accuracy))



def select_best_features(df, model, features, selected=[], accuracy=[]):
    """
    :param df: Dataframe with datetime as index
    :param model: Regression model
    :param features: All the features to test
    :param selected: current best features
    :param accuracy: current accuracy
    :return: Best features for the model
    """
    min_MAPE = 10000
    best_acc = {}
    best_var = ''
    # If no more variables then return
    if len(features) == 0:
        print(accuracy[-1])
        print(selected)
        return selected, accuracy[-1]

    for v in features:
        current = selected.copy()
        current.append(v)
        measures = one_week_test(df, model, current)
        if measures['MAPE'] < min_MAPE:
            min_MAPE = measures['MAPE']
            best_var = v
            best_acc = measures

    # If no improvement then return
    if len(accuracy) != 0 and min_MAPE > accuracy[-1]['MAPE']:
        return selected, accuracy[-1]

    features.remove(best_var)
    selected.append(best_var)
    accuracy.append(best_acc)
    return select_best_features(df, model, features,
                                selected=selected, accuracy=accuracy)


def one_week_test(df, model, features):
    """
    :param df: Dataframe with datetime as index
    :param model: Regression model
    :param features: features to use
    :return: The mean accuracy measures for the last
            week of the dataset
    """
    x = df[features]
    y = df["Consumption(Wh)"]
    dates = [datetime.fromisoformat(d) for d in df.index]

    results = {"MAE": [], "MSE": [], "RMSE": [], "MAPE": []}

    last_date = dates[-1]
    train_last_date = last_date-timedelta(weeks=1)

    x_train = x[:str(train_last_date)]
    y_train = y[:str(train_last_date)]
    x_test = x[str(train_last_date):]
    y_test = y[str(train_last_date):]

    trained_model = model(set=[x_train, y_train, x_test, y_test])

    test_first_date = train_last_date

    for i in range(1, 8):
        end_of_day = test_first_date+timedelta(days=1)
        x_test = x[str(test_first_date):str(end_of_day)]
        y_test = y[str(test_first_date):str(end_of_day)]
        test_first_date = end_of_day
        y_predict = trained_model.predict(x_test)
        aggregated = aggregate(y_test.values, y_predict)
        acc = evaluate_model(aggregated[0], aggregated[1])

        for k in acc.keys():
            results[k].append(acc[k])
    for k in results.keys():
        results[k] = np.mean(results[k])
    return results



def evaluate_model(y, y_pred, show=False):
    """
    :param y: trues values
    :param y_pred: predicted values
    :param show: set True for plots and prints
    :return: Accuracy measures
    """
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


def aggregate(y, y_predict):
    """
    :param y: true values
    :param y_predict: predicted values
    :return: aggregates values hour by hour
    """
    aggregated_y = []
    aggregated_y_pred = []
    index = 0
    while index <= len(y) - 4:
        aggregated_y.append(sum(y[index:index + 4]) / 4)
        aggregated_y_pred.append(sum(y_predict[index:index + 4]) / 4)
        index += 4
    return [aggregated_y, aggregated_y_pred]


if __name__ == '__main__':
    var10 = ['Minutes', 'Snow depth', 'Day', 'Weekend', 'Snowfall']

    features = ["Minutes", "Day", "Week", "Weekend", "Month", "Temperature",
                "Humidity", "Pressure", "Wind speed", "Wind direction", "Snowfall",
                "Snow depth", "Irradiation", "Rainfall", 'Previous_4d_mean_cons']

    df = pd.read_csv('Datasets/10_test.csv', index_col='Datetime')
    last_date = datetime.fromisoformat(df.index[-1])-timedelta(weeks=8)
    df = df['2020-02-15 00:15:00':'2020-08-15 00:15:00']
    #find_models_features(df, features.copy(), '10_test')
    print(one_week_test(df, mlp_model, var10))







