from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import helper
import numpy as np
import pandas as pd


def knn_regressor(set, scale=False, show=False):
    x_train = set[0]
    y_train = set[1]
    x_test = set[2]
    y_test = set[3]

    if scale:
        sc = StandardScaler()
        scaler = sc.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    model = KNeighborsRegressor(
        n_neighbors=60,
        weights='uniform',
        algorithm='auto',
        leaf_size=30,
        metric='minkowski')

    model.fit(x_train, y_train)
    if show:
        y_predict = model.predict(x_test)
        aggregated = helper.aggregate(y_test.values, y_predict)
        helper.plot_model(y_test.values, y_predict, 'KNN')
        helper.plot_model(aggregated[0], aggregated[1], 'KNN_aggregated')
        print("Accuracy : ")
        print(helper.evaluate_model(y_test.values, y_predict))
        print("Accuracy for aggregated values :")
        print(helper.evaluate_model(aggregated[0], aggregated[1]))
    return model


def parameter_search():

    parameters = {'n_neighbors': [20, 50, 60, 75, 100],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto'],
        'leaf_size': [30, 35, 40, 50],
        'metric': ['minkowski']}

    var10 = ['Minutes', 'Snow depth', 'Day', 'Weekend', 'Snowfall']

    df = pd.read_csv('../Datasets/02/10.csv', index_col='Datetime')
    df = df['2020-02-08 00:00:00':'2020-02-08 00:00:00']

    df.reset_index(inplace=True)
    x_train = df[var10]
    y_train = df["Consumption(Wh)"]

    tscv = TimeSeriesSplit(n_splits=5, test_size=672)

    mlp_gs = GridSearchCV(KNeighborsRegressor(), param_grid=parameters, cv=tscv,
                          scoring='neg_root_mean_squared_error')
    mlp_gs.fit(x_train, y_train)
    best_params = mlp_gs.best_params_
    best_score = mlp_gs.best_score_

    with open('knn_gridsearch', 'w') as f:
        f.write(str(best_params) + '\n' + str(best_score))


if __name__ == '__main__':

    variables10 = ['Minutes', 'Month', 'Weekend', 'Temperature', 'Snowfall', 'Pressure']
    df = pd.read_csv('../Datasets/02/10.csv', index_col=["Datetime"],
                     parse_dates=["Datetime"])

    train_set = df['2020-02-08 00:00:00':'2021-01-07 00:00:00']
    test_set = df['2021-01-07 00:00:00':'2021-01-08 00:00:00']

    x_train = np.transpose([train_set[var].to_numpy() for var in variables10])
    y_train = train_set["Consumption(Wh)"]
    x_test = np.transpose([test_set[var].to_numpy() for var in variables10])
    y_test = test_set["Consumption(Wh)"]

    knn_regressor(set=[x_train, y_train, x_test, y_test], show=True, scale=True)

    #parameter_search()