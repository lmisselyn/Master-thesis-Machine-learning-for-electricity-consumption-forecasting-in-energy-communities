import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import helper
from sklearn import metrics


def mlp_model(filename, variables):
    df = pd.read_csv(filename)
    #x = np.transpose([df[var].to_numpy() for var in variables])
    x = df[["Minutes", "Day", "Weekend", "Week", "Month", "Temperature"]]
    y = df["Consumption(Wh)"]
    size = len(df)-96
    x_train = x[:size]
    y_train = y[:size]
    x_test = x[size:]
    y_test = y[size:]

    sc = StandardScaler()
    scaled_x = sc.fit(x_train)
    x_train_scaled = sc.fit(x_train)
    x_test_scaled = sc.fit(x_test)

    model = MLPRegressor(
        hidden_layer_sizes=(250, 500,),
        activation='relu',
        solver='adam',
        alpha=0.0005,
        batch_size='auto',
        learning_rate='adaptive',
        learning_rate_init=0.007,
        power_t=0.5,
        max_iter=500,
        shuffle=True,
        random_state=None,
        tol=0.00005,
        verbose=1,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        n_iter_no_change=10,
        max_fun=15000)

    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)

    helper.evaluate_model(y_test.values, y_predict)
    helper.plot_model(y_test.values, y_predict)

if __name__ == '__main__':
    #best10 = ['Minutes', 'Weekend', 'Temperature', 'Wind direction', 'Wind speed', 'Day of year', 'Day', 'Snowfall', 'Rainfall']
    mlp_model('one_year_10.csv', helper.get_features('one_year_10.csv'))
    mlp_model('one_year_09.csv', helper.get_features('one_year_09.csv'))