import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_regression
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

def plot_model(y, y_predict):
    fig, ax = plt.subplots()
    ax.plot(y, label='True values')
    ax.plot(y_predict, label='Predicted values')
    ax.set_ylabel("Consumption(Wh)")
    ax.legend(facecolor='white')
    plt.show()

def evaluate_model(y, y_pred):
    MAE = metrics.mean_absolute_error(y, y_pred)
    MSE = metrics.mean_squared_error(y, y_pred)
    RMSE = metrics.mean_squared_error(y, y_pred, squared=False)
    MAPE = metrics.mean_absolute_percentage_error(y, y_pred)
    print("Mean absolute error : " + str(MAE))
    print("Mean squared error : " + str(MSE))
    print("Root Mean square error : " + str(RMSE))
    print("Mean absolute percentage error : " + str(MAPE))


def feature_selection(filename):
    df = pd.read_csv(filename)
    x = df.drop(columns=["Date", "Hour", "Index(Wh)", "Consumption(Wh)"])
    y = df["Consumption(Wh)"]
    selection = SelectKBest(f_regression, k=11)
    selection.fit(x, y)
    features = ["Minutes", "Day", "Weekend", "Week", "Month", "Day of year", "Temperature", "Humidity", "Pressure", "Wind speed",
                "Wind direction", "Rainfall", "Snowfall", "Snow depth", "Irradiation"]
    print(selection.get_feature_names_out())

def get_features(filename):
    if filename == "one_year_10.csv":
        return ['Minutes', 'Day', 'Week', 'Weekend', 'Temperature', 'Humidity',
                'Pressure', 'Wind speed', 'Wind direction', 'Irradiation']
    else:
        return ['Minutes', 'Week', 'Month', 'Day of year', 'Temperature',
                'Humidity', 'Wind speed', 'Snowfall', 'Snow depth', 'Irradiation']

if __name__ == '__main__':
    feature_selection("one_year_10.csv")
    feature_selection('one_year_09.csv')