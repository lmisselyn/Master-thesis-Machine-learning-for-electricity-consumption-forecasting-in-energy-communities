import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import pandas as pd
import helper


def SVM_regressor_model(set, scale=False, show=False):
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

    model = svm.SVR()
    model.fit(x_train, y_train)

    if show:
        y_predict = model.predict(x_test)
        aggregated = helper.aggregate(y_test.values, y_predict)
        helper.plot_model(y_test.values, y_predict, 'SVM')
        helper.plot_model(aggregated[0], aggregated[1], 'SVM_aggregated')
        print("Accuracy : ")
        print(helper.evaluate_model(y_test.values, y_predict))
        print("Accuracy for aggregated values :")
        print(helper.evaluate_model(aggregated[0], aggregated[1]))
    return model

if __name__ == '__main__':

    variables10 = ['Minutes', 'Month', 'Weekend', 'Temperature', 'Snowfall', 'Pressure']
    df = pd.read_csv('../Datasets/02/10.csv', index_col=["Datetime"],
                     parse_dates=["Datetime"])

    train_set = df[:'2021-02-05 00:00:00']
    test_set = df['2021-02-05 00:00:00':'2021-02-06 00:00:00']

    x_train = np.transpose([train_set[var].to_numpy() for var in variables10])
    y_train = train_set["Consumption(Wh)"]
    x_test = np.transpose([test_set[var].to_numpy() for var in variables10])
    y_test = test_set["Consumption(Wh)"]
    print("Agrregated accuracy")
    print(SVM_regressor_model(set=[x_train, y_train, x_test, y_test], scale=True))