import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as scipy
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


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
    if data is False:
        df = data
    else:
        df = pd.read_csv(filename)
    correl_df = pd.DataFrame([[0 for i in range(len(variables))]], columns=variables)
    for v in variables:
        x = df[v]
        y = df["Consumption(Wh)"]
        pearson = scipy.pearsonr(x, y)[0]
        correl_df[v][0] = pearson
    if bool:
        print(correl_df.to_string())
    return correl_df


def coeff_correl_manuel(filename, variables, print):
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
    correl_df = pd.DataFrame([[0 for i in range(len(variables))]], columns=variables)
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
        correl_df[var][0] = (numerator / denominator)
    if print:
        print(correl_df.to_string())
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


def make_sets(filename):
    """
    parameters ;
        filename : name of the csv file
    return ;
        training set, validation set, test set
    """
    df = pd.read_csv(filename)
    data = [[], [], []]
    i = 0
    for j in range(len(df)):
        data[i].append(df.loc[j])
        i += 1
        if i > 2:
            i = 0
    train_df = pd.DataFrame(data[0], columns=df.columns)
    validation_df = pd.DataFrame(data[1], columns=df.columns)
    test_df = pd.DataFrame(data[2], columns=df.columns)
    return train_df, validation_df, test_df


def variable_selection(filename, variables):
    train_df, validation_df, test_df = make_sets(filename)
    selected_var = []
    y = train_df["Consumption(Wh)"]
    while len(variables) > 0:
        max_accur = 0
        best_var
        for v in variables:
            y_validation = validation_df["Consumption(Wh)"]
            model = LinearRegression()
            model.fit(train_df[v].to_numpy().reshape(-1, 1), y)
            y_predict = model.predict(validation_df[v].to_numpy().reshape(-1, 1))
            cvrmse = (np.sqrt(mean_squared_error(y_validation, y_predict)))/y_validation.mean()
            MBE = np.mean(y_predict - y_validation)
            R2 = r2_score(y_validation, y_predict)
            accuracy = 0.4*cvrmse + 0.3*MBE + 0.3*R2
            if accuracy > max_accur:
                max_accur = accuracy
                best_var = v
        selected_var.append(best_var)
        variables.remove(best_var)



if __name__ == '__main__':
    variables = ["Day", "Week", "Weekend", "Month", "Temperature", "Humidity", "Pressure",
                 "Wind speed", "Wind direction", "Snowfall", "Snow depth", "Irradiation", "Rainfall"]
    """
    corr_matrice('one_year_10.csv', [])

    coeff_correl('one_year_10.csv', variables)
    regress_visu('one_year_10.csv', variables)
    linear_predict_model('one_year_10.csv', variables)
    
    coeff_correl_manuel('one_year_10.csv', variables)
    coeff_correl('one_year_10.csv', variables)
    """
    variable_selection('one_year_10.csv', variables)