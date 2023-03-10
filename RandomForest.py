from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import helper
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd


def random_forest_model_2(filename, variables):
    train_df, validation_df, test_df = helper.make_sets(filename)
    x = np.transpose([train_df[var].to_numpy() for var in variables])
    y = train_df["Consumption(Wh)"]
    model = RandomForestRegressor(max_features='auto', bootstrap=True, criterion='absolute_error')
    model.fit(x, y)
    x_test = np.transpose([test_df[var].to_numpy() for var in variables])
    y_test = test_df["Consumption(Wh)"]
    y_predict = model.predict(x_test)
    MAE = np.mean(np.abs(y_predict - y_test.values))
    RMSE = np.sqrt(mean_squared_error(y_test.values, y_predict))
    print("Mean absolute error : " + str(MAE))
    print("Root Mean square error : " + str(RMSE))
    helper.plot_model(y_test.values, y_predict)


def random_forest_model(filename=None, set=[], scale=False ):
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
        x = df[["Minutes", "Day", "Weekend", "Week", "Month", "Temperature"]]
        y = df["Consumption(Wh)"]
        # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        size = len(df) - 96
        x_train = x[:size]
        y_train = y[:size]
        x_test = x[size:]
        y_test = y[size:]

    if scale:
        sc = StandardScaler()
        scaler = sc.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    model = RandomForestRegressor(
        n_estimators=100,
        criterion='absolute_error',
        max_depth=150,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features='auto',
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=2,
        random_state=5,
        verbose=2,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None)

    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    aggregated = helper.aggregate(y_test.values, y_predict)
    #helper.plot_model(y_test.values, y_predict)
    #helper.plot_model(aggregated[0], aggregated[1], 'R_F')
    #return helper.evaluate_model(y_test.values, y_predict)
    return helper.evaluate_model(aggregated[0], aggregated[1])



if __name__ == '__main__':
    best10 = ['Minutes', 'Weekend', 'Temperature', 'Wind direction', 'Wind speed', 'Day of year', 'Day', 'Snowfall',
              'Rainfall']
    best09 = ['Minutes', 'Week', 'Temperature', 'Irradiation', 'Pressure', 'Snow depth', 'Month', 'Wind direction',
              'Weekend', 'Day', 'Humidity', 'Wind speed']
    print(random_forest_model('one_year_10.csv'))
    #random_forest_model('one_year_09.csv')
