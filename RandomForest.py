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

def random_forest_model(filename, variables):
    df = pd.read_csv(filename)
    x = np.transpose([df[var].to_numpy() for var in variables])
    y = df["Consumption(Wh)"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    sc = StandardScaler()
    scaler = sc.fit(x_train)
    trainX_scaled = scaler.transform(x_train)
    testX_scaled = scaler.transform(x_test)

    model = RandomForestRegressor(max_features='auto', bootstrap=True, criterion='absolute_error')
    model.fit(trainX_scaled, y_train)

    y_predict = model.predict(testX_scaled)

    helper.evaluate_model(y_test.values, y_predict)
    helper.plot_model(y_test.values, y_predict)


if __name__ == '__main__':

    best10 = ['Minutes', 'Weekend', 'Temperature', 'Wind direction', 'Wind speed', 'Day of year', 'Day', 'Snowfall', 'Rainfall']
    best09 = ['Minutes', 'Week', 'Temperature', 'Irradiation', 'Pressure', 'Snow depth', 'Month', 'Wind direction', 'Weekend', 'Day', 'Humidity', 'Wind speed']
    random_forest_model('one_year_10.csv', helper.get_features('one_year_10.csv'))
    random_forest_model('one_year_09.csv', helper.get_features('one_year_09.csv'))