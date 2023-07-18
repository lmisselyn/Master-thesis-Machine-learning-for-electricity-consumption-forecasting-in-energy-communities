import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn

def time_plot():
    df = pd.read_csv("Datasets/02/one_year_10.csv")
    tot_cons_hour = {}
    nb_hour = {}
    hours = df["Heure"][:96]
    cons_per_hour = []
    for i in range(len(df)):
        hour = df["Heure"][i]
        if hour not in tot_cons_hour:
            tot_cons_hour[hour] = df["Consumption(Wh)"][i]
            nb_hour[hour] = 1
        else:
            tot_cons_hour[hour] += df["Consumption(Wh)"][i]
            nb_hour[hour] += 1

    for h in hours:
        cons_per_hour.append(tot_cons_hour[h] / nb_hour[h])
    cons_per_hour_df = pd.DataFrame({"Hour": hours, "Consumption": cons_per_hour})
    cons_per_hour_df.plot(x="Hour", y="Consumption", grid=True, yticks=(np.arange(100, 400, 25)),
                          ylabel="Average consumption(Wh)", xlabel="Hour of the day")
    plt.xticks(np.arange(0, 96, 4))
    plt.show()


def month_plot():
    df = pd.read_csv("Datasets/02/one_year_10.csv")
    nb_month = [0 for i in range(12)]
    tot_cons_month = [0 for i in range(12)]
    months = ['Ja', 'Fe', 'Ma', 'Ap', 'May', 'Ju', 'Jul', 'Au', 'Se', 'Oc', 'No', 'De']

    for i in range(len(df)):
        month = int(df["Date"][i][3:5])
        tot_cons_month[month - 1] += df["Consumption(Wh)"][i]
        nb_month[month - 1] += 1

    for m in range(len(tot_cons_month)):
        tot_cons_month[m] = tot_cons_month[m] / nb_month[m]
    cons_per_month_df = pd.DataFrame({"Month": months, "Consumption": tot_cons_month})
    cons_per_month_df.plot(x="Month", y="Consumption", grid=True, yticks=np.arange(180, 300, 10),
                           ylabel="Average consumption(Wh)", xlabel="Month of the year")
    plt.xticks(np.arange(12), months)
    plt.show()

def irradiation_plot():
    df = pd.read_csv("Datasets/02/one_year_10.csv")
    new_x = np.arange(0, 2300, 10)
    print(new_x)
    irrad = df["Irradiation"]
    cons = df["Consumption(Wh)"]
    x_irrad = []
    y_cons = []
    dict = {}
    nb_var = {}
    for i in range(len(irrad)):
        if irrad[i] in dict:
            dict[irrad[i]] += cons[i]
            nb_var[irrad[i]] += 1
        else:
            dict[irrad[i]] = cons[i]
            nb_var[irrad[i]] = 1
    for k in dict.keys():
        x_irrad.append(k)
        y_cons.append(dict[k]/nb_var[k])
    plt.scatter(x_irrad, y_cons)
    plt.xlabel =("Irradiation")
    plt.ylabel=("Consumption(Wh)")
    plt.show()

def temp_plot():
    df = pd.read_csv("Datasets/02/one_year_10.csv")
    tot_cons_temp = {}
    nb_temp = {}
    cons_per_temp = []
    for i in range(len(df)):
        temp = (df["Temperature"][i]) - 273
        if temp not in tot_cons_temp:
            tot_cons_temp[temp] = df["Consumption(Wh)"][i]
            nb_temp[temp] = 1
        else:
            tot_cons_temp[temp] += df["Consumption(Wh)"][i]
            nb_temp[temp] += 1
    for k in tot_cons_temp.keys():
        cons_per_temp.append(tot_cons_temp[k]/nb_temp[k])
    Temperature = tot_cons_temp.keys()
    cons_per_temp_df = pd.DataFrame({"Temperature": Temperature, "Consumption": cons_per_temp})
    cons_per_temp_df.plot(x="Temperature", y="Consumption", grid=True)
    plt.show()

def correlation_plot(filename):
    var = ['Consumption(Wh)', 'Week', 'Month', 'Day_of_year', 'Day', 'Minutes',
           'Weekend', 'temperature_2m', 'relativehumidity_2m',
           'dewpoint_2m', 'apparent_temperature', 'pressure_msl',
           'surface_pressure', 'snowfall', 'weathercode', 'cloudcover',
           'cloudcover_low', 'cloudcover_mid', 'cloudcover_high',
           'shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
           'direct_normal_irradiance', 'windspeed_10m', 'winddirection_10m',
           'Prev_4d_mean_cons', 'Prev_4w_mean_cons', 'precipitation']
    df = pd.read_csv(filename)
    df = df[var]
    m = df.corr(method='spearman')
    m = m['Consumption(Wh)'].copy()
    m.sort_values(inplace=True, key=abs)
    print(m)
    #plt.figure(figsize=(10, 6))
    #seaborn.heatmap(m)
    #plt.show()


if __name__ == '__main__':
    for i in ['01', '02', '03', '04', '05', '06', '07', '08']:
        filename = 'Datasets/' + i + '/' + i + 'final.csv'
        print(i)
        correlation_plot(filename)

