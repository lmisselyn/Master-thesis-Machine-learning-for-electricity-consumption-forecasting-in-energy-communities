import pandas as pd
import datetime


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

    # compute conso
    one_year_df['Consumption(Wh)'] = 0
    for i in range(1, len(one_year_df)):
        indice = len(one_year_df) - i
        one_year_df.at[indice, 'Consumption(Wh)'] = \
            (one_year_df.at[indice, "Index(Wh)"] - one_year_df.at[indice - 1, "Index(Wh)"])
    one_year_df.at[0, 'Consumption(Wh)'] = 94

    # add columns : Day, Week, Weekend, Month
    new_colunms = ["Day of year", "Day", "Week", "Weekend", "Month"]
    Day_y = []
    Day = []
    Week = []
    Weekend = []
    Month = []
    for s in new_colunms:
        one_year_df[s] = 0
    first_d = datetime.date(2020, 2, 8)
    for i in range(len(one_year_df)):
        d = one_year_df.at[i, "Date"]
        date = datetime.date(int(d[6:]), int(d[3:5]), int(d[:2]))
        Day_y.append(date.toordinal() - first_d.toordinal())
        Day.append(date.isoweekday())
        Month.append(date.month)
        if date.weekday() == 5 or date.weekday() == 6 :
            Weekend.append(1)
        else : Weekend.append(0)
        Week.append(Day_y[i] // 7)
    one_year_df["Day of year"] = Day_y
    one_year_df["Day"] = Day
    one_year_df["Week"] = Week
    one_year_df["Weekend"] = Weekend
    one_year_df["Month"] = Month
    # get weather data
    df = pd.read_csv("../weather_data.csv")
    one_year_w = df.loc[59231:94366]
    one_year_w_df = pd.DataFrame(one_year_w, columns=df.columns)
    one_year_w_df.drop(columns=["# Date", "UT time"], inplace=True)
    one_year_w_df.reset_index(drop=True, inplace=True)
    one_year_w_df.rename(columns={"Relative Humidity": "Humidity", "Short-wave irradiation,": "Irradiation"},
                         inplace=True)

    for i in range(len(one_year_w_df)):
        val = one_year_w_df.at[i, "Irradiation"]
        val = format_irradiation(val)
        one_year_w_df.at[i, "Irradiation"] = float(val)

    df_concat = pd.concat([one_year_df, one_year_w_df], axis=1)
    df_concat.to_csv("one_year_10.csv")

def format_irradiation(val):
    new_val = val.replace(",", "", 1)
    if new_val.count(".") > 1:
        new_val = new_val.replace(".", "", 1)
    return new_val


if __name__ == '__main__':
    data_get_csv()
