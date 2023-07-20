from Models.linear_regression import *

if __name__ == '__main__':
    #model = l

    first_d = {'01': '2020-02-25 00:00:00', '02': '2020-02-15 00:00:00', '03': '2020-02-27 00:00:00',
               '04': '2020-07-24 00:00:00',
               '05': '2020-08-22 00:00:00', '06': '2020-08-25 00:00:00', '07': '2020-08-25 00:00:00',
               '08': '2020-10-06 00:00:00'}

    pearson = {'01': ['apparent_temperature', 'diffuse_radiation', 'dewpoint_2m',
       'Prev_4w_mean_cons', 'Prev_4d_mean_cons'], '02': ['relativehumidity_2m', 'Day', 'Weekend', 'Prev_4d_mean_cons',
       'Prev_4w_mean_cons'], '03': ['shortwave_radiation', 'diffuse_radiation', 'Minutes',
       'Prev_4d_mean_cons', 'Prev_4w_mean_cons'], '04': ['Day', 'Weekend', 'Minutes', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'],
       '05': ['temperature_2m', 'apparent_temperature', 'dewpoint_2m',
       'Prev_4d_mean_cons', 'Prev_4w_mean_cons'], '06': ['apparent_temperature', 'Day', 'windspeed_10m', 'Prev_4d_mean_cons',
       'Prev_4w_mean_cons'], '07': ['diffuse_radiation', 'apparent_temperature', 'temperature_2m',
       'Prev_4d_mean_cons', 'Prev_4w_mean_cons'], '08': ['apparent_temperature', 'temperature_2m', 'Minutes',
       'Prev_4d_mean_cons', 'Prev_4w_mean_cons']}

    spearman = {'01': ['shortwave_radiation', 'direct_normal_irradiance', 'dewpoint_2m',
       'Prev_4w_mean_cons', 'Prev_4d_mean_cons'], '02': ['Weekend', 'apparent_temperature', 'temperature_2m',
       'Prev_4d_mean_cons', 'Prev_4w_mean_cons'], '03': ['apparent_temperature', 'dewpoint_2m', 'Minutes', 'Prev_4d_mean_cons',
       'Prev_4w_mean_cons'], '04': ['direct_radiation', 'diffuse_radiation', 'shortwave_radiation',
       'Prev_4d_mean_cons', 'Prev_4w_mean_cons'], '05': ['apparent_temperature', 'dewpoint_2m', 'Minutes', 'Prev_4d_mean_cons',
       'Prev_4w_mean_cons'], '06': ['direct_normal_irradiance', 'Day', 'windspeed_10m', 'Prev_4w_mean_cons',
       'Prev_4d_mean_cons'], '07': ['direct_normal_irradiance', 'Prev_4d_mean_cons', 'shortwave_radiation',
       'diffuse_radiation', 'Prev_4w_mean_cons'], '08': ['diffuse_radiation', 'relativehumidity_2m', 'Minutes',
       'Prev_4w_mean_cons', 'Prev_4d_mean_cons']
}

    mutual_i = {'01': ['Minutes', 'temperature_2m', 'dewpoint_2m', 'apparent_temperature',
       'shortwave_radiation', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'], '02': ['Minutes', 'temperature_2m', 'relativehumidity_2m', 'dewpoint_2m',
       'apparent_temperature', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'], '03': ['Minutes', 'dewpoint_2m', 'apparent_temperature', 'diffuse_radiation',
       'direct_normal_irradiance', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'], '04': ['Minutes', 'temperature_2m', 'dewpoint_2m', 'apparent_temperature',
       'shortwave_radiation', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'], '05': ['Minutes', 'temperature_2m', 'dewpoint_2m', 'apparent_temperature',
       'diffuse_radiation', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'], '06' : ['Day', 'shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
       'windspeed_10m', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'], '07': ['Minutes', 'temperature_2m', 'dewpoint_2m', 'apparent_temperature',
       'direct_normal_irradiance', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons'], '08': ['Minutes', 'temperature_2m', 'dewpoint_2m', 'apparent_temperature',
       'winddirection_10m', 'Prev_4d_mean_cons', 'Prev_4w_mean_cons']}

    wrapp_r2 = {'01':{}}

    wrapp_mape = {}

    for i in ['01', '02', '03', '04', '05', '06', '07', '08']:
        filename = 'Datasets/' + i + '/' + i + 'final.csv'
        first_date = first_d[i]
        features = pearson[i]
