from sklearn.ensemble import RandomForestRegressor
import helper
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

def random_forest_model(filename, variables):
    train_df, validation_df, test_df = helper.make_sets(filename)
    x = np.transpose([train_df[var].to_numpy() for var in variables])
    y = train_df["Consumption(Wh)"]
    model = RandomForestRegressor(max_features='auto', bootstrap=True, criterion='absolute_error')
    model.fit(x, y)
    x_test = np.transpose([test_df[var].to_numpy() for var in variables])
    y_test = test_df["Consumption(Wh)"]
    y_predict = model.predict(x_test)
    MAE = np.mean(np.abs(y_predict - y_test))
    RMSE = np.sqrt(mean_squared_error(y_test, y_predict))
    print("Mean absolute error : " + str(MAE))
    print("Root Mean square error : " + str(RMSE))
    df = pd.DataFrame({"true values": y_test[:50], "predicted values": y_predict[:50]})
    print(df)

if __name__ == '__main__':
    best10 = ['Minutes', 'Weekend', 'Temperature', 'Wind direction', 'Wind speed', 'Day of year', 'Day', 'Snowfall', 'Rainfall']
    best09 = ['Minutes', 'Week', 'Temperature', 'Irradiation', 'Pressure', 'Snow depth', 'Month', 'Wind direction', 'Weekend', 'Day', 'Humidity', 'Wind speed']
    random_forest_model('one_year_09.csv', best09)