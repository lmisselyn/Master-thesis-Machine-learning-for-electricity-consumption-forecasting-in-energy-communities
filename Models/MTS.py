import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR

def mts_model(filename, variables):
    df = pd.read_csv('../Datasets/10/one_year_10_datetime.csv', parse_dates=['Datetime'], index_col=['Datetime'])
    train_set = df[:0.8*(len(df))]
    test_set = df[0.8*(len(df)):]
    x_train = train_set.drop(columns=["Date", "Hour", "Consumption(Wh)"])
    y_train = train_set["Consumption(Wh)"]
    x_test = test_set.drop(columns=["Date", "Hour", "Consumption(Wh)"])
    y_test = test_set["Consumption(Wh)"]

    model = VAR(endog=x_train)
    model_fit = model.fit()


if __name__ == '__main__':
    best10 = ['Minutes', 'Weekend', 'Temperature', 'Wind direction',
              'Wind speed', 'Day of year', 'Day', 'Snowfall', 'Rainfall']
    mts_model('../Datasets/10/one_year_10_datetime.csv', best10)

