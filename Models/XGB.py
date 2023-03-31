import pandas as pd
import xgboost
from sklearn.preprocessing import StandardScaler
import helper
import numpy as np

#from sklearn.model_selection import GridSearchCV

def XGB_regressor_model(filename=None, set=[], scale=False):
    """
    train a random forest model with the dataset 'filename'
    - set (optional) : provide train and test sets
    - scale (boolean) : scale data if true
    """
    if len(set) != 0:
        x_train = set[0]
        y_train = set[1]
        x_test = set[2]
        y_test = set[3]

    else:
        df = pd.read_csv(filename)
        x = df[['Minutes', 'Snow depth', 'Day', 'Weekend', 'Snowfall']]
        y = df["Consumption(Wh)"]
        # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        size = len(df) - 288
        x_train = x[:size]
        y_train = y[:size]
        x_test = x[size:]
        y_test = y[size:]

    if scale:
        sc = StandardScaler()
        scaler = sc.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    model = xgboost.XGBRegressor(
        booster='gbtree',
        eval_metric='rmse',
        early_stopping_rounds=100,
        objective='reg:linear',
        learning_rate=0.015,
        max_depth=6,
        n_estimators=1500
    )

    model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], verbose=100)
    y_predict = model.predict(x_test)
    aggregated = helper.aggregate(y_test.values, y_predict)
    helper.plot_model(test_set.index, y_test.values, y_predict, 'R_F')
    helper.plot_model(test_set.index, aggregated[0], aggregated[1], 'R_F_aggregated')
    print("Accuracy : ")
    print(helper.evaluate_model(y_test.values, y_predict))
    return helper.evaluate_model(aggregated[0], aggregated[1])


if __name__ == '__main__':
    variables09 = ['Minutes', 'Humidity', 'Day', 'Snow depth', 'Snowfall', 'Weekend', 'Temperature']
    df = pd.read_csv('../Datasets/one_year_09_datetime.csv', index_col=["Datetime"],
                             parse_dates=["Datetime"])

    train_set = df['2021-04-28 00:00:00':'2021-05-28 00:00:00']
    test_set = df['2021-05-28 00:00:00':]

    x_train = np.transpose([train_set[var].to_numpy() for var in variables09])
    y_train = train_set["Consumption(Wh)"]
    x_test = np.transpose([test_set[var].to_numpy() for var in variables09])
    y_test = test_set["Consumption(Wh)"]
    print("Agrregated accuracy")
    print(XGB_regressor_model(set=[x_train, y_train, x_test, y_test], scale=True))

    variables10 = ['Minutes', 'Month', 'Weekend', 'Temperature', 'Snowfall', 'Pressure']
    df = pd.read_csv('../Datasets/one_year_10_datetime.csv', index_col=["Datetime"],
                             parse_dates=["Datetime"])

    train_set = df['2021-1-05 00:00:00':'2021-02-05 00:00:00']
    test_set = df['2021-02-05 00:00:00':]

    x_train = np.transpose([train_set[var].to_numpy() for var in variables09])
    y_train = train_set["Consumption(Wh)"]
    x_test = np.transpose([test_set[var].to_numpy() for var in variables09])
    y_test = test_set["Consumption(Wh)"]
    print("Agrregated accuracy")
    print(XGB_regressor_model(set=[x_train, y_train, x_test, y_test], scale=True))

