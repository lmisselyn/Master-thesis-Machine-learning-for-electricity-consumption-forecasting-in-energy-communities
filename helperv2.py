from datetime import datetime, timedelta
import numpy as np

import helper
from helper import aggregate, evaluate_model
from sklearn import metrics
from Models.KNN import knn_regressor
from Models.RandomForest import random_forest_model
from Models.SVM import SVM_regressor_model
from Models.XGB import XGB_regressor_model
from Models.mlp_regression import mlp_model
from Models.linear_regression import linear_regression

models = {"SVM": SVM_regressor_model, "XGB": XGB_regressor_model,
          'R_F': random_forest_model, "MLP": mlp_model}


def select_best_model(dataset):
    best_acc = 100
    best_model = 0
    """
    Select the best model for the specific dataset
    """
    for model in models.keys():
        with open('Features/' + model + '_' + dataset + '.txt', 'r') as file:
            line = file.readlines()[1]
            acc = line.split(':')
            mape = float(acc[-1].strip()[:6])
            if mape < best_acc:
                best_acc = mape
                best_model = model
    return best_model


def find_models_features(df, features, dataset, train_n_weeks):
    """
    Find the best features for each model and store results in a text file
    """
    for model in models.keys():
        selected = []
        accuracy = []
        select_best_features(df, models[model], features.copy(), train_n_weeks, selected, accuracy)
        result = ''
        for s in selected:
            result += s + " "

        with open('Features/' + model + '_' + dataset + '.txt', 'w') as file:
            file.write(result + '\n' + str(accuracy))


def get_feature(model, dataset):
    """
    Retrieve the best features for a specific model and dataset
    """
    with open('Features/' + model + '_' + dataset + '.txt', 'r') as file:
        features = file.readline().split(' ')[:-1]
        return features


def select_best_features(df, model, features, train_n_weeks, selected=[], accuracy=[]):
    """
    :param df: Dataframe with datetime as index
    :param model: Regression model
    :param features: All the features to test
    :param selected: current best features
    :param accuracy: current accuracy
    :return: Best features for the model
    """
    min_MAPE = float('inf')
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
        # measures = multi_month_test(df, model, current, train_n_weeks)
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
    print(selected)
    print(accuracy[-1])
    return select_best_features(df, model, features, train_n_weeks,
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
    train_last_date = last_date - timedelta(weeks=1)

    x_train = x[:str(train_last_date)]
    y_train = y[:str(train_last_date)]
    x_test = x[str(train_last_date):]
    y_test = y[str(train_last_date):]

    trained_model = model(set=[x_train, y_train, x_test, y_test])

    test_first_date = train_last_date

    for i in range(1, 8):
        end_of_day = test_first_date + timedelta(days=1)
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


def multi_month_test(df, model, features, train_n_weeks):
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
    train_first_date = dates[0]

    while train_first_date + timedelta(weeks=train_n_weeks) < last_date:

        train_last_date = train_first_date + timedelta(weeks=train_n_weeks)
        test_last_date = train_last_date + timedelta(days=1)
        x_train = x[str(train_first_date):str(train_last_date)]
        y_train = y[str(train_first_date):str(train_last_date)]
        x_test = x[str(train_last_date):str(test_last_date)]
        y_test = y[str(train_last_date):str(test_last_date)]

        trained_model = model(set=[x_train, y_train, x_test, y_test])
        y_predict = trained_model.predict(x_test)
        aggregated = aggregate(y_test.values, y_predict)
        acc = evaluate_model(aggregated[0], aggregated[1])

        for k in acc.keys():
            results[k].append(acc[k])
        train_first_date = train_last_date

    for k in results.keys():
        results[k] = np.mean(results[k])
    return results
