import pandas as pd
import xgboost
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

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
        eval_metric='rmse')

    param_grid = {"max_depth": [5, 6],
                  "n_estimators": [100, 200, 300],
                  "learning_rate": [0.01, 0.013, 0.015, 0.017]}

    search = GridSearchCV(model, param_grid, cv=5).fit(x_train, y_train)

    print("The best hyperparameters are ", search.best_params_)

    '''
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    aggregated = helper.aggregate(y_test.values, y_predict)
    # helper.plot_model(y_test.values, y_predict)
    # helper.plot_model(aggregated[0], aggregated[1], 'R_F')
    # return helper.evaluate_model(y_test.values, y_predict)
    return helper.evaluate_model(aggregated[0], aggregated[1])
    '''

if __name__ == '__main__':
    XGB_regressor_model(filename='../Datasets/one_year_09.csv')

