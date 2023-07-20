import pandas as pd
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

first_d = {'01': '2020-02-25 00:00:00', '02': '2020-02-15 00:00:00', '03': '2020-02-27 00:00:00',
           '04': '2020-07-24 00:00:00',
           '05': '2020-08-22 00:00:00', '06': '2020-08-25 00:00:00', '07': '2020-08-25 00:00:00',
           '08': '2020-10-06 00:00:00'}

models = [LinearRegression(),
          KNeighborsRegressor(
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
              max_samples=None)]


def mutual_info(filename, features):
    df = pd.read_csv(filename)
    y = df['Consumption(Wh)']
    x = df[features]
    select = SelectKBest(mutual_info_regression, k=5)
    select.fit(x, y)
    mask = select.get_support()
    print(x.columns[mask])


def wrapping_feature_selection(filename, model, features, score):
    df = pd.read_csv(filename)
    x = df[features]
    y = df['Consumption(Wh)']
    sfs = SFS(model,
              n_features_to_select='auto',
              direction='forward',
              scoring=score,
              cv=None)
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
           'Weekend', 'temperature_2m', 'relativehumidity_2m',
           'dewpoint_2m', 'apparent_temperature',
           'shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
           'direct_normal_irradiance', 'windspeed_10m',
           'Prev_4d_mean_cons', 'Prev_4w_mean_cons']

    #for m in models[:3]:
        #print(m)
    for i in ['01', '02', '03', '04']: #, '05', '06', '07', '08']:
        filename = 'Datasets/' + i + '/' + i + 'final.csv'
        print(i)
            #wrapping_feature_selection(filename, m, var, 'r2')
        mutual_info(filename, var)
        # print(correlation(filename, 'pearson', 5, var))
