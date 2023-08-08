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
        y_cons.append(dict[k] / nb_var[k])
    plt.scatter(x_irrad, y_cons)
    plt.xlabel = ("Irradiation")
    plt.ylabel = ("Consumption(Wh)")
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
        cons_per_temp.append(tot_cons_temp[k] / nb_temp[k])
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
    # plt.figure(figsize=(10, 6))
    # seaborn.heatmap(m)
    # plt.show()


if __name__ == '__main__':
    ano2_with = [0.288, 0.2471, 0.27382, 0.304, 0.497789, 0.2939, 0.16363, 0.19592, 0.2243, 0.22473, 0.310, 0.22589,
            0.216839, 0.236, 0.2192, 0.215, 0.338, 0.3848, 0.39, 0.3377, 0.33763, 0.3089, 0.288, 0.42359, 0.3134,
            0.289, 0.383, 0.3771, 0.28832, 0.2841, 0.31381, 0.3127, 0.335, 0.3672, 0.3129, 0.38773, 0.35, 0.32,
            0.311, 0.295, 0.3582, 0.49928, 0.3599, 0.3295, 0.3191, 0.352, 0.36, 0.309, 0.35894, 0.4177, 0.4129,
            0.5231, 0.266, 0.2686, 0.2679, 0.4349, 0.422, 0.34, 0.3463, 0.3496, 0.6587, 0.557, 0.588, 1.558, 0.499,
            0.5277, 0.40, 0.45438, 0.452, 0.463, 0.4525, 0.442, 0.44, 0.45, 0.456, 0.424, 0.420, 0.4383, 0.40, 0.4697,
            0.40, 0.39, 0.452, 0.413, 0.449, 0.40, 0.395, 0.386, 0.3968, 0.397, 0.437, 0.3, 0.411, 0.375, 0.282,
            0.14, 0.34, 0.48575, 0.54277, 1.217, 0.45188, 0.2397, 0.2466]  # Average MAPE: 0.3846907815085709

    ano2_without = [0.288, 0.2471, 0.27382, 0.304, 0.497789, 0.2939, 0.16363, 0.19592, 0.2243, 0.22473, 0.310, 0.22589,
                    0.216839, 0.236, 0.2192, 0.215, 0.338, 0.3848, 0.39, 0.3377, 0.33763, 0.3089, 0.288, 0.42359,
                    0.3134, 0.289, 0.383, 0.3771, 0.28832, 0.2841, 0.31381, 0.3127, 0.335, 0.3672, 0.3129, 0.38773,
                    0.35, 0.32, 0.311, 0.295, 0.3582, 0.49928, 0.3599, 0.3295, 0.3191, 0.352, 0.36, 0.309, 0.35894,
                    0.4177, 0.4129, 0.5231, 0.2781, 0.250646, 0.2598, 0.375133, 0.5734, 0.4383, 0.40343, 0.48823,
                    0.3634, 0.3628, 0.417615, 3.146, 0.5577, 0.4987, 0.5449, 0.6656, 0.601342, 0.50, 0.7775,
                    0.74452, 0.5895, 0.7384, 0.667, 0.68, 0.524, 0.569, 0.67912, 0.7085, 0.771613, 0.63407, 0.5855,
                    0.55137, 0.494968, 0.49964, 0.4951, 0.347624, 0.31657, 0.35, 0.4477, 0.309, 0.342889, 0.514,
                    0.205, 0.35619, 0.336435, 0.64655, 0.18744, 0.213547, 0.256726, 0.167952, 0.111]

    print(len(ano2_with))
    print(len(ano2_without))
    plt.figure().set_figwidth(8)
    plt.plot(np.arange(len(ano2_without)), ano2_with, label='with anomaly detection')
    plt.plot(np.arange(len(ano2_without)), ano2_without, label='without anomaly detection')
    plt.plot(np.arange(len(ano2_without)), np.full(len(ano2_with), 0.4), linewidth=0.5, label='threshold')
    plt.legend()
    plt.title("Anomaly detection algorithm visualisation - dataset 01 & 08")
    #plt.xticks(np.arange(len(ano2_without), 2))
    plt.ylabel('MAPE')
    plt.xlabel('Week')
    plt.show()