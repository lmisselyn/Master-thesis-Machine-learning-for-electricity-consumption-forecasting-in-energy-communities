import xgboost

first_d = {'01': '2020-02-25 00:00:00', '02': '2020-02-15 00:00:00', '03': '2020-02-27 00:00:00',
           '04': '2020-07-24 00:00:00',
           '05': '2020-08-22 00:00:00', '06': '2020-08-25 00:00:00', '07': '2020-08-25 00:00:00',
           '08': '2020-10-06 00:00:00'}

best_model = {'01': xgboost.XGBRegressor(booster='gbtree',
                                         eval_metric='rmse',
                                         early_stopping_rounds=20,
                                         objective='reg:squarederror',
                                         learning_rate=0.0085,  # best 0.0075
                                         max_depth=4,  # best 6
                                         n_estimators=165,  # best 125
                                         # subsample=0.85,
                                         colsample_bylevel=0.5),
              '02': xgboost.XGBRegressor(booster='gbtree',
                                         eval_metric='rmse',
                                         early_stopping_rounds=20,
                                         objective='reg:squarederror',
                                         learning_rate=0.008,  # best 0.0075
                                         max_depth=4,  # best 6
                                         n_estimators=145,  # best 125
                                         # subsample=0.85,
                                         colsample_bylevel=0.5),
              '03': xgboost.XGBRegressor(booster='gbtree',
                                         eval_metric='rmse',
                                         early_stopping_rounds=20,
                                         objective='reg:squarederror',
                                         learning_rate=0.0087,  # best 0.0075
                                         max_depth=4,  # best 6
                                         n_estimators=60,  # best 125
                                         subsample=0.85,
                                         colsample_bylevel=0.2),
              '04': xgboost.XGBRegressor(booster='gbtree',
                                         eval_metric='rmse',
                                         early_stopping_rounds=20,
                                         objective='reg:squarederror',
                                         learning_rate=0.005,  # best 0.0075
                                         max_depth=1,  # best 6
                                         n_estimators=110,  # best 125
                                         # subsample=0.85,
                                         colsample_bylevel=0.2)
              }


pearson = {'01': ['apparent_temperature', 'diffuse_radiation', 'dewpoint_2m',
                  'Prev_4w_mean_cons', 'Prev_4d_mean_cons'],
           '02': ['relativehumidity_2m', 'Day', 'Weekend', 'Prev_4d_mean_cons',
                  'Prev_4w_mean_cons'], '03': ['shortwave_radiation', 'diffuse_radiation', 'Minutes',
                                               'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
           '04': ['Day', 'Weekend', 'Minutes', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
           '05': ['temperature_2m', 'apparent_temperature', 'dewpoint_2m',
                  'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
           '06': ['apparent_temperature', 'Day', 'windspeed_10m', 'Prev_4d_mean_cons',
                  'Prev_4w_mean_cons'], '07': ['diffuse_radiation', 'apparent_temperature', 'temperature_2m',
                                               'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
           '08': ['apparent_temperature', 'temperature_2m', 'Minutes',
                  'Prev_4d_mean_cons', 'Prev_4w_mean_cons']}

spearman = {'01': ['shortwave_radiation', 'direct_normal_irradiance', 'dewpoint_2m',
                   'Prev_4w_mean_cons', 'Prev_4d_mean_cons'],
            '02': ['Weekend', 'apparent_temperature', 'temperature_2m',
                   'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
            '03': ['apparent_temperature', 'dewpoint_2m', 'Minutes', 'Prev_4d_mean_cons',
                   'Prev_4w_mean_cons'], '04': ['direct_radiation', 'diffuse_radiation', 'shortwave_radiation',
                                                'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
            '05': ['apparent_temperature', 'dewpoint_2m', 'Minutes', 'Prev_4d_mean_cons',
                   'Prev_4w_mean_cons'], '06': ['direct_normal_irradiance', 'Day', 'windspeed_10m', 'Prev_4w_mean_cons',
                                                'Prev_4d_mean_cons'],
            '07': ['direct_normal_irradiance', 'Prev_4d_mean_cons', 'shortwave_radiation',
                   'diffuse_radiation', 'Prev_4w_mean_cons'],
            '08': ['diffuse_radiation', 'relativehumidity_2m', 'Minutes',
                   'Prev_4w_mean_cons', 'Prev_4d_mean_cons']
            }

mutual_i = {'01': ['Minutes', 'dewpoint_2m', 'apparent_temperature', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
            '02': ['temperature_2m', 'dewpoint_2m', 'apparent_temperature', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
            '03': ['Minutes', 'dewpoint_2m', 'shortwave_radiation', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
            '04': ['Minutes', 'temperature_2m', 'apparent_temperature', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
            '05': ['Minutes', 'dewpoint_2m', 'apparent_temperature', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
            '06': ['shortwave_radiation', 'diffuse_radiation', 'windspeed_10m', 'Prev_4d_mean_cons',
                   'Prev_4w_mean_cons'],
            '07': ['Minutes', 'temperature_2m', 'dewpoint_2m', 'apparent_temperature', 'Prev_4w_mean_cons'],
            '08': ['Minutes', 'temperature_2m', 'dewpoint_2m', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons']}


wrapp_r2 = {'LR': {'01': ['Minutes', 'apparent_temperature', 'direct_radiation', 'direct_normal_irradiance',
                          'windspeed_10m', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
                   '02': ['Day', 'Weekend', 'relativehumidity_2m', 'direct_radiation', 'diffuse_radiation',
                          'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
                   '03': ['Day', 'Minutes', 'Weekend', 'relativehumidity_2m', 'diffuse_radiation',
                          'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
                   '04': ['Minutes', 'Weekend', 'shortwave_radiation', 'direct_radiation',
                          'diffuse_radiation', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
                   '05': ['Day', 'Weekend', 'shortwave_radiation', 'diffuse_radiation',
                          'direct_normal_irradiance', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
                   '06': ['Weekend', 'relativehumidity_2m', 'dewpoint_2m', 'diffuse_radiation',
                          'direct_normal_irradiance', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
                   '07': ['dewpoint_2m', 'apparent_temperature', 'diffuse_radiation',
                          'direct_normal_irradiance', 'windspeed_10m', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
                   '08': ['Minutes', 'temperature_2m', 'relativehumidity_2m', 'apparent_temperature',
                          'direct_normal_irradiance', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons']},
            'SVM': {'01': ['Day', 'Weekend', 'apparent_temperature', 'windspeed_10m', 'Prev_4d_mean_cons'],
                    '02': ['relativehumidity_2m', 'dewpoint_2m', 'apparent_temperature', 'Prev_4d_mean_cons',
                           'Prev_4w_mean_cons'],
                    '03': ['Day', 'Weekend', 'dewpoint_2m', 'windspeed_10m', 'Prev_4w_mean_cons'],
                    '04': ['Day', 'Weekend', 'windspeed_10m', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
                    '05': ['Day', 'Weekend', 'dewpoint_2m', 'windspeed_10m', 'Prev_4w_mean_cons'],
                    '06': ['Day', 'relativehumidity_2m', 'shortwave_radiation', 'Prev_4d_mean_cons',
                           'Prev_4w_mean_cons'],
                    '07': ['Day', 'apparent_temperature', 'shortwave_radiation', 'Prev_4d_mean_cons',
                           'Prev_4w_mean_cons'],
                    '08': ['relativehumidity_2m', 'apparent_temperature', 'windspeed_10m', 'Prev_4d_mean_cons',
                           'Prev_4w_mean_cons']},
            'RF': {'01': ['Minutes', 'Weekend', 'dewpoint_2m', 'shortwave_radiation', 'Prev_4d_mean_cons'],
                   '02': ['Minutes', 'dewpoint_2m', 'apparent_temperature', 'Prev_4d_mean_cons',
                          'Prev_4w_mean_cons'],
                   '03': ['Day', 'Minutes', 'shortwave_radiation', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
                   '04': ['Day', 'Minutes', 'shortwave_radiation', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
                   '05': ['Day', 'Minutes', 'Weekend', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
                   '06': ['apparent_temperature', 'shortwave_radiation', 'windspeed_10m', 'Prev_4d_mean_cons',
                          'Prev_4w_mean_cons'],
                   '07': ['Day', 'Weekend', 'relativehumidity_2m', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
                   '08': ['Day', 'Minutes', 'relativehumidity_2m', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons']},
            'XGB': {'01': ['Day', 'Minutes', 'Weekend', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
                    '02': ['Minutes', 'Weekend', 'relativehumidity_2m', 'shortwave_radiation', 'Prev_4w_mean_cons'],
                    '03': ['Day', 'Minutes', 'Weekend', 'shortwave_radiation', 'Prev_4w_mean_cons'],
                    '04': ['Day', 'Minutes', 'Weekend', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
                    '05': ['Day', 'Minutes', 'shortwave_radiation', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
                    '06': ['relativehumidity_2m', 'shortwave_radiation', 'windspeed_10m', 'Prev_4d_mean_cons',
                           'Prev_4w_mean_cons'],
                    '07': ['relativehumidity_2m', 'shortwave_radiation', 'windspeed_10m', 'Prev_4d_mean_cons',
                           'Prev_4w_mean_cons'],
                    '08': ['Minutes', 'Weekend', 'apparent_temperature', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons']}}