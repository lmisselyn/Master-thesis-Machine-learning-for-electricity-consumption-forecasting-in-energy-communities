from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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
        n_neighbors=50,
        weights='distance',
        algorithm='auto',
        leaf_size=100, p=2,
        metric='minkowski',
        metric_params=None,
        n_jobs=None)

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


if __name__ == '__main__':

    variables10 = ['Minutes', 'Month', 'Weekend', 'Temperature', 'Snowfall', 'Pressure']
    df = pd.read_csv('../Datasets/10_test.csv', index_col=["Datetime"],
                             parse_dates=["Datetime"])

    train_set = df[:'2021-02-05 00:00:00']
    test_set = df['2021-02-05 00:00:00':'2021-02-06 00:00:00']

    x_train = np.transpose([train_set[var].to_numpy() for var in variables10])
    y_train = train_set["Consumption(Wh)"]
    x_test = np.transpose([test_set[var].to_numpy() for var in variables10])
    y_test = test_set["Consumption(Wh)"]
    print("Agrregated accuracy")
    print(knn_regressor(set=[x_train, y_train, x_test, y_test], scale=True))