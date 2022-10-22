import pandas as pd
import matplotlib.pyplot as plt

def data_get_csv():
    df = pd.read_csv("../donneeconso10.csv")[["Date", "Heure", "Index(Wh)"]]
    one_year = df.loc[34:35296]
    one_year_df = pd.DataFrame(one_year, columns=df.columns)
    one_year_df.reset_index(drop=True, inplace=True)

    #suppress bugs
    for i in range(len(one_year_df)):
        hour = int(one_year_df.at[i, "Heure"][3:])
        if hour not in [0, 15, 30, 45]:
            one_year_df.drop(index=i, inplace=True)

    one_year_df.reset_index(drop=True, inplace=True)
    for i in range(1, len(one_year_df)):
        indice = len(one_year_df)-i
        one_year_df.at[indice, "Index(Wh)"] -= one_year_df.at[indice-1, "Index(Wh)"]
    one_year_df.at[0, "Index(Wh)"] = 94
    one_year_df.to_csv("one_year10.csv")

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
        cons_per_hour.append(tot_cons_hour[h]/nb_hour[h])
    cons_per_hour_df = pd.DataFrame({"Hour": hours, "Consommation": cons_per_hour})
    cons_per_hour_df.plot(x="Hour", y="Consommation")
    plt.show()


if __name__ == '__main__':
    time_plot()