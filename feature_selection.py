import numpy as np
import pandas as pd
import xgboost
from sklearn.model_selection import TimeSeriesSplit
import global_variables
from datetime import datetime, timedelta
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor

models = [xgboost.XGBRegressor(booster='gbtree',
                               eval_metric='rmse',
                               # early_stopping_rounds=20,
                               objective='reg:squarederror',
                               learning_rate=0.008,  # best 0.0075
                               max_depth=4,  # best 6
                               n_estimators=135,  # best 125
                               subsample=0.95,
                               colsample_bylevel=0.5),
          SVR(C=5,
              epsilon=0.4,
              gamma='scale'),
          RandomForestRegressor(n_estimators=150,
                                criterion='absolute_error',
                                max_depth=6,
                                min_samples_split=2,
                                max_features=0.5,
                                max_leaf_nodes=None,
                                random_state=5,
                                min_impurity_decrease=0.0,
                                bootstrap=True,  # True
                                n_jobs=2,
                                max_samples=0.85,
                                )]


def mutual_info(filename, features):
    df = pd.read_csv(filename)
    y = df['Consumption(Wh)']
    x = df[features]
    select = SelectKBest(mutual_info_regression, k=5)
    select.fit(x, y)
    mask = select.get_support()
    print(x.columns[mask])


def wrapping_feature_selection(df, model, features, score):
    x = df[features]
    y = df['Consumption(Wh)']

    tscv = TimeSeriesSplit(n_splits=4, max_train_size=10752, test_size=672)
    sfs = SFS(model,
              n_features_to_select='auto',
              direction='forward',
              scoring=score,
              cv=tscv)
    sfs.fit(x, y)
    print(sfs.get_feature_names_out())


def correlation(filename, method, k, features):
    df = pd.read_csv(filename)
    df = df[features]
    m = df.corr(method=method)
    m = m['Consumption(Wh)'].copy()
    m.sort_values(inplace=True, key=abs)

    return m.keys()[-k - 1:-1]


if __name__ == '__main__':
    var = ['Day', 'Minutes',
           'Weekend', 'relativehumidity_2m',
           'dewpoint_2m', 'apparent_temperature',
           'shortwave_radiation',
           'windspeed_10m',
           'Prev_4d_mean_cons', 'Prev_4w_mean_cons']

    for m in models:
        print(m)
        for i in ['01', '02', '03', '04', '05', '06', '07', '08']:
            filename = 'Datasets/' + i + '/' + i + 'final.csv'
            df = pd.read_csv(filename, index_col='Datetime')
            train_first_date = datetime.fromisoformat(global_variables.first_d[i])
            df = df[str(train_first_date):]
            print(i)
            wrapping_feature_selection(df, m, var, 'neg_mean_absolute_percentage_error')
