import pandas as pd
import xgboost
from sklearn.preprocessing import StandardScaler
import helper
import numpy as np

#from sklearn.model_selection import GridSearchCV

def XGB_regressor_model(set, scale=False, show=False):
    """
    train a random forest model with the dataset 'filename'
    - set (optional) : provide train and test sets
    - scale (boolean) : scale data if true
    """
    x_train = set[0]
    y_train = set[1]
    x_test = set[2]
    y_test = set[3]

    if scale:
        sc = StandardScaler()
        scaler = sc.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    model = xgboost.XGBRegressor(
        booster='gbtree',
        eval_metric='rmse',
        early_stopping_rounds=100,
        objective='reg:squarederror',
        learning_rate=0.015, #best 0.015
        max_depth=None, #best None
        n_estimators=100 #best 100
    )

    model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], verbose=False)

    if show:
        y_predict = model.predict(x_test)
        aggregated = helper.aggregate(y_test.values, y_predict)
        helper.plot_model(y_test.values, y_predict, 'XGB')
        helper.plot_model(aggregated[0], aggregated[1], 'XGB_aggregated')
        print("Accuracy : ")
        print(helper.evaluate_model(y_test.values, y_predict))
        print("Accuracy for aggregated values : ")
        print(helper.evaluate_model(aggregated[0], aggregated[1]))

    return model


if __name__ == '__main__':

    variables10 = ['Minutes', 'Month', 'Weekend', 'Temperature', 'Snowfall', 'Pressure']

    var10 = ['Previous_4d_mean_cons', 'Snow depth', 'Weekend', 'Irradiation', 'Minutes', 'Week',
             'Wind direction', 'Month', 'Snowfall', 'Temperature', 'Rainfall']

    df = pd.read_csv('../Datasets/10_test.csv', index_col=["Datetime"],
                             parse_dates=["Datetime"])


    #date = datetime.fromisoformat()
    train_set = df['2020-02-16 00:00:00':'2021-01-07 00:00:00']
    test_set = df['2021-01-15 00:00:00':'2021-01-16 00:00:00']

    x_train = np.transpose([train_set[var].to_numpy() for var in variables10])
    y_train = train_set["Consumption(Wh)"]
    x_test = np.transpose([test_set[var].to_numpy() for var in variables10])
    y_test = test_set["Consumption(Wh)"]

    XGB_regressor_model(set=[x_train, y_train, x_test, y_test], show=True, scale=True)

