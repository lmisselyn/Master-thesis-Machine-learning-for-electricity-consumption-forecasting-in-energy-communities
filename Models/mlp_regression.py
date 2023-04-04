import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import helper
from sklearn import metrics


def mlp_model(set, scale=False, show=False):
    """
    train a random multi layer perceptron model with the dataset 'filename'
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
        verbose=0,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=True,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        n_iter_no_change=10,
        max_fun=1500)

    model.fit(x_train, y_train)
    if show:
        y_predict = model.predict(x_test)
        aggregated = helper.aggregate(y_test.values, y_predict)
        helper.plot_model(y_test.values, y_predict, 'mlp')
        helper.plot_model(aggregated[0], aggregated[1], 'mlp_aggregated')
        print('Accuracy :')
        print(helper.evaluate_model(y_test.values, y_predict))
        print("Aggregated accuracy :")
        print(helper.evaluate_model(aggregated[0], aggregated[1]))
    return model


if __name__ == '__main__':

    variables10 = ['Minutes', 'Snow depth', 'Day', 'Weekend', 'Snowfall']
    var10 = ['Previous_4d_mean_cons', 'Snow depth', 'Weekend', 'Irradiation', 'Minutes', 'Week',
             'Wind direction', 'Month', 'Snowfall', 'Temperature', 'Rainfall']
    df = pd.read_csv('../Datasets/10_test.csv', index_col=["Datetime"],
                             parse_dates=["Datetime"])

    train_set = df['2020-02-16 00:00:00':'2021-02-05 00:00:00']
    test_set = df['2021-02-05 00:00:00':'2021-02-06 00:00:00']

    x_train = np.transpose([train_set[var].to_numpy() for var in var10])
    y_train = train_set["Consumption(Wh)"]
    x_test = np.transpose([test_set[var].to_numpy() for var in var10])
    y_test = test_set["Consumption(Wh)"]

    mlp_model(set=[x_train, y_train, x_test, y_test], scale=True, show=True)