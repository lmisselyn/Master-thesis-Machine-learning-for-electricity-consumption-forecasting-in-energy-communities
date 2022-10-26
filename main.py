import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def data_get_csv():
    df = pd.read_csv("../donneeconso10.csv")[["Date", "Heure", "Index(Wh)"]]
    one_year = df.loc[34:35199]

    one_year_df = pd.DataFrame(one_year, columns=df.columns)
    one_year_df.reset_index(drop=True, inplace=True)

    # suppress bugs
    for i in range(len(one_year_df)):
        hour = int(one_year_df.at[i, "Heure"][3:])
        if hour not in [0, 15, 30, 45]:
            one_year_df.drop(index=i, inplace=True)

    one_year_df.reset_index(drop=True, inplace=True)
    for i in range(1, len(one_year_df)):
        indice = len(one_year_df) - i
        one_year_df.at[indice, "Index(Wh)"] -= one_year_df.at[indice - 1, "Index(Wh)"]
    one_year_df.at[0, "Index(Wh)"] = 94

    df = pd.read_csv("../weather_data.csv")
    one_year_w = df.loc[59231:94366]

    one_year_w_df = pd.DataFrame(one_year_w, columns=df.columns)
    one_year_w_df.drop(columns=["# Date", "UT time", "Pressure",
                                "Rainfall", "Snowfall", "Snow depth"], inplace=True)
    one_year_w_df.reset_index(drop=True, inplace=True)
    one_year_w_df.rename(columns={"Relative Humidity": "Humidity", "Short-wave irradiation,": "Irradiation"},
                         inplace=True)

    for i in range(len(one_year_w_df)):
        val = one_year_w_df.at[i, "Irradiation"]
        val = format_irradiation(val)
        one_year_w_df.at[i, "Irradiation"] = float(val)

    df_concat = pd.concat([one_year_df, one_year_w_df], axis=1)
    print(df_concat)
    df_concat.to_csv("one_year_10.csv")


def time_plot():
    df = pd.read_csv("one_year_10.csv")
    tot_cons_hour = {}
    nb_hour = {}
    hours = df["Heure"][:96]
    cons_per_hour = []
    for i in range(len(df)):
        hour = df["Heure"][i]
        if hour not in tot_cons_hour:
            tot_cons_hour[hour] = df["Index(Wh)"][i]
            nb_hour[hour] = 1
        else:
            tot_cons_hour[hour] += df["Index(Wh)"][i]
            nb_hour[hour] += 1

    for h in hours:
        cons_per_hour.append(tot_cons_hour[h] / nb_hour[h])
    cons_per_hour_df = pd.DataFrame({"Hour": hours, "Consumption": cons_per_hour})
    cons_per_hour_df.plot(x="Hour", y="Consumption", grid=True, yticks=(np.arange(100, 400, 25)),
                          ylabel="Average consumption(Wh)", xlabel="Hour of the day")
    plt.xticks(np.arange(0, 96, 4))
    plt.show()


def month_plot():
    df = pd.read_csv("one_year_10.csv")
    nb_month = [0 for i in range(12)]
    tot_cons_month = [0 for i in range(12)]
    months = ['Ja', 'Fe', 'Ma', 'Ap', 'May', 'Ju', 'Jul', 'Au', 'Se', 'Oc', 'No', 'De']

    for i in range(len(df)):
        month = int(df["Date"][i][3:5])
        tot_cons_month[month - 1] += df["Index(Wh)"][i]
        nb_month[month - 1] += 1

    for m in range(len(tot_cons_month)):
        tot_cons_month[m] = tot_cons_month[m] / nb_month[m]
    cons_per_month_df = pd.DataFrame({"Month": months, "Consumption": tot_cons_month})
    cons_per_month_df.plot(x="Month", y="Consumption", grid=True, yticks=np.arange(180, 300, 10),
                           ylabel="Average consumption(Wh)", xlabel="Month of the year")
    plt.xticks(np.arange(12), months)
    plt.show()


def irradiation_plot():
    df = pd.read_csv("one_year_10.csv")
    new_x = np.arange(0, 2300, 10)
    print(new_x)
    irrad = df["Irradiation"]
    cons = df["Index(Wh)"]
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

def format_irradiation(val):
    new_val = val.replace(",", "", 1)
    if new_val.count(".") > 1:
        new_val = new_val.replace(".", "", 1)
    return new_val

def temp_plot():
    df = pd.read_csv("one_year_10.csv")
    tot_cons_temp = {}
    nb_temp = {}
    cons_per_temp = []
    for i in range(len(df)):
        temp = (df["Temperature"][i]) - 273
        if temp not in tot_cons_temp:
            tot_cons_temp[temp] = df["Index(Wh)"][i]
            nb_temp[temp] = 1
        else:
            tot_cons_temp[temp] += df["Index(Wh)"][i]
            nb_temp[temp] += 1
    for k in tot_cons_temp.keys():
        cons_per_temp.append(tot_cons_temp[k]/nb_temp[k])
    Temperature = tot_cons_temp.keys()[1:len]
    cons_per_temp_df = pd.DataFrame({"Temperature":  , "Consumption": cons_per_temp})
    cons_per_temp_df.plot(x="Temperature", y="Consumption", grid=True)
    plt.show()

if __name__ == '__main__':
    time_plot()
    month_plot()
    irradiation_plot()
    temp_plot()
