import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


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
    df_concat = pd.concat([one_year_df, one_year_w_df], axis=1)
    #colums_name = one_year_df.columns.append(one_year_w_df.columns)
    #df_concat.rename(columns=colums_name)
    print(df_concat)
    df_concat.to_csv("one_year_10.csv")
    # one_year_df.to_csv("one_year10.csv")
    # print(len(one_year_df))
    # print(len(one_year_w))


def time_plot():
    df = pd.read_csv("one_year10.csv")
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

    plt.show()


def month_plot():
    df = pd.read_csv("one_year10.csv")
    dict_month = {}
    nb_month = [0 for i in range(12)]
    tot_cons_month = [0 for i in range(12)]
    months = ['Ja', 'Fe', 'Ma', 'Ap', 'May', 'Ju', 'Jul', 'Au', 'Se', 'Oc', 'No', 'De']

    date = df["Date"]
    for i in range(len(df)):
        month = int(df["Date"][i][3:5])
        tot_cons_month[month - 1] += df["Index(Wh)"][i]
        nb_month[month - 1] += 1

    for m in range(len(tot_cons_month)):
        tot_cons_month[m] = tot_cons_month[m] / nb_month[m]
    cons_per_month_df = pd.DataFrame({"Month": months, "Consumption": tot_cons_month})
    cons_per_month_df.plot(x="Month", y="Consumption", grid=True)
    plt.show()


if __name__ == '__main__':
    data_get_csv()
