import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as scipy
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def corr_matrice(filename, col_to_drop):
    df = pd.read_csv(filename)
    data = df.drop(columns=col_to_drop)
    matrice_corr = data.corr().round(1)
    sns.heatmap(data=matrice_corr, annot=True)
    plt.show()


def coeff_correl(filename, variables):
    df = pd.read_csv(filename)
    for v in variables:
        x = df[v]
        y = df["Consumption(Wh)"]
        pearson = scipy.pearsonr(x, y)
        spearman = scipy.spearmanr(x, y)
        kendall = scipy.kendalltau(x, y)
        print("Correlation coefficient for " + v)
        print("pearson :" + str(round(pearson[0], 3)))
        print("spearman :" + str(round(spearman[0], 3)))
        print("kendall : " + str(round(kendall[0], 3)))
        print('\n')


def regress_visu(filename, variables):
    df = pd.read_csv(filename)
    for v in variables:
        x = df[v]
        y = df["Consumption(Wh)"]
        slope, intercept, r, p, stderr = scipy.stats.linregress(x, y)
        fig, ax = plt.subplots()
        ax.plot(x, y, linewidth=0, marker='s', label='Data points')
        ax.plot(x, intercept + slope * x, label='linear_regress')
        ax.set_xlabel(v)
        ax.set_ylabel("Consumption(Wh)")
        ax.legend(facecolor='white')
        plt.show()


def linear_predict_model(filename, variables):
    df = pd.read_csv(filename)
    X_train, X_test, Y_train, Y_test = train_test_split(df[variables],
                                                        df["Consumption(Wh)"],
                                                        test_size=0.2,
                                                        random_state=5)
    model = LinearRegression()
    model.fit(X_train, Y_train)

    y_train_predict = model.predict(X_train)
    rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
    r2 = r2_score(Y_train, y_train_predict)
    print('Training')
    print('--------------------------------------')
    print('Mean squre error : {}'.format(rmse))
    print('R2 score : {}'.format(r2))
    print('\n')
    # model evaluation for testing set
    y_test_predict = model.predict(X_test)
    rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
    r2 = r2_score(Y_test, y_test_predict)
    print('Testing')
    print('--------------------------------------')
    print('Mean squre error {}'.format(rmse))
    print('R2 score {}'.format(r2))


if __name__ == '__main__':
    corr_matrice('one_year_10.csv', [])
    variables = ["Day", "Week", "Weekend", "Month", "Temperature", "Humidity", "Pressure",
                "Wind speed", "Wind direction", "Snowfall", "Snow depth", "Irradiation", "Rainfall"]
    coeff_correl('one_year_10.csv', variables)
    regress_visu('one_year_10.csv', variables)
    linear_predict_model('one_year_10.csv', variables)

