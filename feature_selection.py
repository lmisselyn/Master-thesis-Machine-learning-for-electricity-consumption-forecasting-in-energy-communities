import pandas as pd
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

models = [KNeighborsRegressor(
        n_neighbors=60,
        weights='uniform',
        algorithm='auto',
        leaf_size=30,
        metric='minkowski'),
        MLPRegressor(
        hidden_layer_sizes=(100, 100, 100),
        activation='relu',
        solver='adam',
        learning_rate='adaptive',
        learning_rate_init=0.005,
        max_iter=1500,
        early_stopping=True,
        validation_fraction=0.1),
        SVR(),
        RandomForestRegressor(
        n_estimators=100,
        criterion='absolute_error',
        max_depth=150,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=2,
        random_state=5,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None),
        XGBRegressor(
        booster='gbtree',
        eval_metric='rmse',
        early_stopping_rounds=100,
        objective='reg:squarederror',
        learning_rate=0.015,
        n_estimators=100)]

def mutual_info(filename, features):
    df = pd.read_csv(filename)
    y = df['Consumption(Wh)']
    x = df[features]
    select = SelectKBest(mutual_info_regression, k=5)
    select.fit(x, y)
    mask = select.get_support()
    print(x.columns[mask])


def wrapping_feature_selection(filename, model, features):
    df = pd.read_csv(filename)
    x = df[features]
    y = df['Consumption(Wh)']
    sfs = SFS(model,
              k_features=7,
              forward=True,
              floating=False,
              scoring='r2',
              cv=0)
    sfs.fit(x, y)
    print(sfs.k_feature_names_)


def correlation(filename, method, features):
    df = pd.read_csv(filename)
    df = df[features]
    m = df.corr(method='spearman')
    m = m['Consumption(Wh)'].copy()
    m.sort_values(inplace=True, key=abs)
    print(m)


if __name__ == '__main__':
    var = ['Day', 'Minutes',
           'Weekend', 'temperature_2m', 'relativehumidity_2m',
           'dewpoint_2m', 'apparent_temperature',
           'shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
           'direct_normal_irradiance', 'windspeed_10m', 'winddirection_10m',
           'Prev_4d_mean_cons', 'Prev_4w_mean_cons', 'precipitation']

    for i in ['01', '02', '03', '04']: # '05', '06', '07', '08']:
        filename = 'Datasets/' + i + '/' + i + 'final.csv'
        print(i)
        for m in models:
            print(m)
            wrapping_feature_selection(filename, m, var)
