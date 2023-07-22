import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as scipy
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from feature_selection import *
import helper


def corr_matrix(filename, col_to_drop):
    """
    parameters ;
        filename : name of the csv file
        col_to_drop : name of colunms not to show in the correlation matrix
    """
    df = pd.read_csv(filename)
    data = df.drop(columns=col_to_drop)
    matrice_corr = data.corr().round(1)
    sns.heatmap(data=matrice_corr, annot=True)
    plt.show()


def coeff_correl(variables, filename=False, bool=False, data=False):
    """
    parameters ;
        filename : name of the csv file
        variables : name of columns for which we compute the pearson coefficient
                    pearson(x=var, y=Consumption)
        print : bollean to know if we have to print result
        data : dataframe to use instead of filename
    return ; Dataframe with pearson coefficient for each column where
            pearson(x=var, y=Consumption)
    """
    df = 0
    if data is not False:
        df = data
    else:
        df = pd.read_csv(filename)
    correl = {}
    for v in variables:
        x = df[v]
        y = df["Consumption(Wh)"]
        pearson = scipy.pearsonr(x, y)[0]
        correl[v] = np.round(pearson, 3)
    correl_df = pd.DataFrame([correl])
    if bool:
        print("Correlation coefficients computed using scipy function\n" + correl_df.to_string())
    return correl_df


def coeff_correl_manuel(filename, variables, bool=False):
    """
    parameters ;
        filename : name of the csv file
        variables : name of columns for which we compute the pearson coefficient
                    pearson(x=var, y=Consumption)
        print : bollean to know if we have to print result
    return ; Dataframe with pearson coefficient for each column where
            pearson(x=var, y=Consumption)
    """
    df = pd.read_csv(filename)
    correl = {}
    y = df["Consumption(Wh)"]
    y_mean = y.mean()
    for var in variables:
        x = df[var]
        x_mean = x.mean()
        numerator = 0
        sum_x = 0
        sum_y = 0
        for i in range(len(y)):
            numerator += (x[i] - x_mean) * (y[i] - y_mean)
            sum_x += np.power((x[i] - x_mean), 2)
            sum_y += np.power((y[i] - y_mean), 2)
        denominator = np.sqrt(sum_x * sum_y)
        correl[var] = np.round((numerator / denominator), 3)
    correl_df = pd.DataFrame([correl])
    if bool:
        print("Correlation coefficients computed manually\n" + correl_df.to_string())
    return correl_df


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


def linear_regression(set, scale=False, show=False):
    """
    train a linear model with the training set provided in set[0]
    and test it on the testing set provided in
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

    model = LinearRegression()
    model.fit(x_train, y_train)

    if show:
        y_predict = model.predict(x_test)
        aggregated = helper.aggregate(y_test.values, y_predict)
        helper.plot_model(y_test.values, y_predict, 'linear regression')
        helper.plot_model(aggregated[0], aggregated[1], 'Linear regression - dataset01 - (2021-02-27) ')
        print("Accuracy : ")
        print(helper.evaluate_model(y_test.values, y_predict))
        print("Accuracy for aggregated values : ")
        print(helper.evaluate_model(aggregated[0], aggregated[1]))

    return model


if __name__ == '__main__':
    var = ['Consumption(Wh)', 'Day', 'Minutes',
           'Weekend', 'temperature_2m', 'relativehumidity_2m',
           'dewpoint_2m', 'apparent_temperature',
           'shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
           'direct_normal_irradiance', 'windspeed_10m', 'winddirection_10m',
           'Prev_4d_mean_cons', 'Prev_4w_mean_cons', 'precipitation']

    for i in ['01']: #, '02', '03', '04', '05', '06', '07', '08']:  #
        filename = '../Datasets/' + i + '/' + i + 'final.csv'
        features = correlation(filename, 'spearman', 5, var)
        df = pd.read_csv(filename, index_col='Datetime')

        train_set = df['2020-11-24 00:00:00':'2021-02-24 00:00:00']
        test_set = df['2021-02-27 00:00:00':'2021-02-28 00:00:00']
        train_visu = df['2021-02-21 00:00:00':'2021-02-24 00:00:00']

        x_train = np.transpose([train_set[var].to_numpy() for var in features])
        y_train = train_set["Consumption(Wh)"]
        x_test = np.transpose([test_set[var].to_numpy() for var in features])
        y_test = test_set["Consumption(Wh)"]
        lr = linear_regression(set=[x_train, y_train, x_test, y_test], show=True)
        print(features)
        print(lr.intercept_)
        print(lr.coef_)

        x_train_visu = train_visu[features]
        y_train_visu = train_visu['Consumption(Wh)']

        plt.plot(y_train_visu,  label='Training data')
        plt.plot(lr.predict(x_train_visu),  label='fitted model')
        plt.title("Lr visualisation - dataset01 - (2021-02-21, 2021-02-24)")
        plt.xticks([''])
        plt.legend()
        plt.ylabel("Consumption(Wh)")
        plt.show()