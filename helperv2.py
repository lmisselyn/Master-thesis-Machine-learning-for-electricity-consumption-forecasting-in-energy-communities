from datetime import datetime, timedelta
import numpy as np
from sklearn import metrics
from Models import XGB, RandomForest, MTS


def select_best_model(df, features):
    models = [XGB.XGB_regressor_model,
                  RandomForest.random_forest_model,
                  MTS.mts_model]
    for model in models:
        select_best_features(df, model, features)



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
        ret = [var for var in selected]
        return ret

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
        print(accuracy[-1])
        print(selected)
        ret = [var for var in selected]
        return ret

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