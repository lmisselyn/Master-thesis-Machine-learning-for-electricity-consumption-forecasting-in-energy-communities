import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as scipy
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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


def plot_model(y, y_predict):
    fig, ax = plt.subplots()
    ax.plot(y, label='True values')
    ax.plot(y_predict, label='Predicted values')
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
    print('Root Mean squre error : {}'.format(rmse))
    print('R2 score : {}'.format(r2))
    print('\n')
    # model evaluation for testing set
    y_test_predict = model.predict(X_test)
    rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
    r2 = r2_score(Y_test, y_test_predict)
    print('Testing')
    print('--------------------------------------')
    print('Root Mean squre error {}'.format(rmse))
    print('R2 score {}'.format(r2))

def final_model(filename, variables):
    train_df, validation_df, test_df = helper.make_sets(filename)
    x = np.transpose([train_df[var].to_numpy() for var in variables])
    y = train_df["Consumption(Wh)"]
    model = LinearRegression()
    model.fit(x, y)
    x_test = np.transpose([test_df[var].to_numpy() for var in variables])
    y_test = test_df["Consumption(Wh)"]
    y_predict = model.predict(x_test)
    helper.evaluate_model(y_test.values, y_predict)
    helper.plot_model(y_test.values, y_predict)



def variable_selection(filename, variables):
    train_df, validation_df, test_df = helper.make_sets(filename)
    selected_var = []
    best_accuracy = -10000
    y = train_df["Consumption(Wh)"]
    iter = 0
    while len(variables) > 0:
        max_accur = -10000
        best_var = ""
        for v in variables:
            y_validation = validation_df["Consumption(Wh)"]
            model = LinearRegression()
            x = []
            x_validation = []
            if len(selected_var) == 0:
                x = train_df[v].to_numpy().reshape(-1, 1)
                x_validation = validation_df[v].to_numpy().reshape(-1, 1)
            else:
                x = [train_df[var].to_numpy() for var in selected_var]
                x.append(train_df[v].to_numpy())
                x_validation = [validation_df[var].to_numpy() for var in selected_var]
                x_validation.append(validation_df[v].to_numpy())
                x = np.array(x).transpose()
                x_validation = np.array(x_validation).transpose()
            model.fit(x, y)
            y_predict = model.predict(x_validation)
            #cvrmse = (np.sqrt(mean_squared_error(y_validation, y_predict))) / y_validation.mean()
            #MBE = np.mean(y_predict - y_validation)
            #MSE = mean_squared_error(y_validation, y_predict)
            MAE = np.mean(np.abs(y_predict - y_validation))
            R2 = model.score(x_validation, y_validation)
            #accuracy = 0.6 * (1 - cvrmse) + 0.4 * (1 - MBE)
            accuracy = -MAE
            if accuracy > max_accur:
                max_accur = accuracy
                best_var = v

        # return if no variable selected
        if best_var != "":
            selected_var.append(best_var)
            variables.remove(best_var)
        else:
            return selected_var, best_accuracy
        print("Iteration " + str(iter) + ":" + str(best_accuracy))
        iter += 1
        if max_accur > best_accuracy:
            if best_accuracy != -10000 and max_accur-best_accuracy < 0.01:
                best_accuracy = max_accur
                return selected_var, best_accuracy
            best_accuracy = max_accur

    return selected_var, best_accuracy


if __name__ == '__main__':

    final_model('one_year_10.csv', helper.get_features('one_year_10.csv'))
    final_model('one_year_09.csv', helper.get_features('one_year_09.csv'))
