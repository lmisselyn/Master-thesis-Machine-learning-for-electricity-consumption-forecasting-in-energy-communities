import random

import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

def get_data_csv_10():
    df = pd.read_csv("../02final.csv")[["Date", "Heure", "Index(Wh)"]]
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
        if date.weekday() == 5 or date.weekday() == 6:
            Weekend.append(1)
        else:
            Weekend.append(0)
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
        rain_val = format_rainfall(one_year_w_df.at[i, "Rainfall"])
        one_year_w_df.at[i, "Rainfall"] = float(rain_val)
    df_concat = pd.concat([one_year_df, one_year_w_df], axis=1)
    df_concat.to_csv("one_year_10.csv")


def get_data_csv_09():
    df = pd.read_csv('Datasets/01/one_year_09.csv')
    """
    weather_df = pd.read_csv('../weather_data.csv')
    one_year_w = weather_df.loc[70175:105214]
    one_year_w_df = pd.DataFrame(one_year_w, columns=weather_df.columns)
    df['Irradiation'] = one_year_w_df['Short-wave irradiation,'].values
    df['Pressure'] = one_year_w_df['Pressure'].values
    df.reset_index(inplace=True)
    """
    for i in range(len(df)):
        if type(df.at[i, "Rainfall"]) == type('str'):
            df.at[i, "Rainfall"] = float(format_rainfall(df.at[i, "Rainfall"]))
    df.to_csv('one_year_09.csv')


def remove_dupplicates(filename):
    df = pd.read_csv(filename)
    df.drop_duplicates(subset=['Datetime'], inplace=True)
    df.to_csv(filename)


def change_hour(filename):
    df = pd.read_csv(filename, index_col=0)
    new_hour = []
    for i in range(len(df)):
        h = df.at[i, "Hour"]
        new_hour.append(format_hour(h))
    df["Minutes"] = new_hour
    df.to_csv(filename)


def format_irradiation(val):
    new_val = val.replace(",", "", 1)
    if new_val.count(".") > 1:
        new_val = new_val.replace(".", "", 1)
    return new_val


def format_rainfall(val):
    cnt = val.count(",")
    if cnt > 0:
        val = val.replace(",", ".", cnt)
    if val.count(".") > 1:
        return val.replace(".", "", 2)
    return val


def format_hour(val):
    minutes = 0
    minutes += 60 * int(val[:2])
    minutes += int(val[3:])
    return minutes


def at_home_feature(filename):
    df = pd.read_csv(filename)
    mean_comsumptiom = df["Consumption(Wh)"].mean()
    print(mean_comsumptiom)
    at_home = []
    for i in range(len(df)):
        c = df.at[i, "Consumption(Wh)"]
        if c > mean_comsumptiom:
            at_home.append(1)
        else:
            at_home.append(0)
    df["AtHome"] = at_home
    df.to_csv("one_year_10_datetime.csv")


def day_of_year_feature(filename):
    df = pd.read_csv(filename)
    doy = []
    for i in range(len(df)):
        date = df.at[i, "Date"]
        doy.append(day_of_year(date))
    df["Day of year"] = doy
    df.to_csv(filename)


def day_of_year(date):
    d = int(date[:2])
    m = int(date[3:5])
    y = int(date[6:])
    if (y % 4 == 0 and y % 100 != 0) or y % 400 == 0:
        return (0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366)[m - 1] + d
    else:
        return (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365)[m - 1] + d


def datetime_format(filename):
    df = pd.read_csv(filename)
    new_datetime = []
    for i in range(len(df)):
        date = df.at[i, "Date"] + ' ' + df.at[i, "Hour"]
        new_datetime.append(datetime.strptime(date, "%d/%m/%Y %H:%M"))
    df["Datetime"] = new_datetime
    df.to_csv(filename)


def mean_array(arr):
    sum = 0
    for item in arr:
        if not isinstance(item, float):
            print(item)
        sum = sum + item
    if len(arr) > 0:
        sum = round(sum / len(arr), 4)
    return sum


def mean_cons_by_week(filename):
    mean_cons = []
    df = pd.read_csv(filename, index_col=['Datetime'])
    dates = [datetime.fromisoformat(d) for d in df.index]
    first_date = dates[0]
    dates.reverse()

    for date in dates:
        tmp = []
        for i in range(1, 5):
            prev_week = date - timedelta(weeks=i)
            if prev_week > first_date and prev_week in dates:
                tmp2 = [df.at[str(prev_week), 'Consumption(Wh)']]
                if len(tmp2) > 1:
                    print(prev_week)
                before = prev_week - timedelta(minutes=15)
                after = prev_week + timedelta(minutes=15)
                for around in [before, after]:
                    if around in dates:
                        tmp2.append(df.at[str(around), 'Consumption(Wh)'])
                    else:
                        print(around)
                        break
                tmp.append(np.mean(tmp2))
        try:
            m = mean_array(tmp)
            mean_cons.append(float(m))
        except:
            print(mean_array(tmp))
    mean_cons.reverse()
    df['Prev_4w_mean_cons'] = mean_cons
    df.to_csv(filename)


def past_consumption_feature(filename, timestep):
    new_feature = []
    df = pd.read_csv(filename, index_col=['Datetime'])
    dates = [datetime.fromisoformat(d) for d in df.index]
    first_date = dates[0]
    dates.reverse()

    for date in dates:
        past_cons = []
        for t in timestep:
            prev_step = date - timedelta(days=t)
            if prev_step > first_date:
                before = str(prev_step - timedelta(minutes=15))
                after = str(prev_step + timedelta(minutes=15))
                tmp = [df.at[before, 'Consumption(Wh)'],
                       df.at[str(prev_step), 'Consumption(Wh)'],
                       df.at[after, 'Consumption(Wh)']]
                past_cons.append(np.mean(tmp))
        new_feature.append(np.mean(past_cons))
    new_feature.reverse()
    df['new_feature'] = new_feature
    df.to_csv(filename)


def mean_cons_by_hour(filename):
    mean_cons = []
    df = pd.read_csv(filename, index_col=['Datetime'])
    dates = [datetime.fromisoformat(d) for d in df.index]
    first_date = dates[0]
    dates.reverse()

    for date in dates:
        tmp = []
        for i in range(3, 7):
            prev_week = date - timedelta(days=i)
            if prev_week > first_date and prev_week in dates:
                tmp2 = [df.at[str(prev_week), 'Consumption(Wh)']]
                if len(tmp2) > 1:
                    print(prev_week)
                before = prev_week - timedelta(minutes=15)
                after = prev_week + timedelta(minutes=15)
                for around in [before, after]:
                    if around in dates:
                        tmp2.append(df.at[str(around), 'Consumption(Wh)'])
                    else:
                        print(around)
                        break
                tmp.append(np.mean(tmp2))
        try:
            m = mean_array(tmp)
            mean_cons.append(float(m))
        except:
            print(mean_array(tmp))
    mean_cons.reverse()
    df['Prev_4d_mean_cons'] = mean_cons
    df.to_csv(filename)


def tmp_date(filename):
    df = pd.read_csv(filename)
    dt = df["Datetime"]
    new_datetime = [datetime.strptime(d, "%d/%m/%Y %H:%M") for d in dt]
    df["Datetime"] = new_datetime
    df.to_csv(filename)


def tmp_find_zero(filename):
    df = pd.read_csv(filename)
    index = df['Index(Wh)']
    for i in range(len(index) - 1):
        if index[i] == index[i + 1]:
            print(i)
            break


def tmp_cons_calcu(filename):
    consumption = []
    consumption.append(0)
    df = pd.read_csv(filename)
    index = df['Index(Wh)']
    for i in range(1, len(index)):
        consumption.append(index[i] - index[i - 1])
    df['Consumption(Wh)'] = consumption
    df.to_csv(filename)


def tmp_time_features(filename):
    df = pd.read_csv(filename)
    dt = df["Datetime"]
    dates = [datetime.strptime(d, "%d/%m/%Y %H:%M") for d in dt]

    week = []
    month = []
    doy = []
    dow = []
    weekend = []
    minute = []
    for d in dates:
        week.append(d.isocalendar().week)
        month.append(d.month)
        doy.append(d.timetuple().tm_yday)
        dow.append(d.isocalendar().weekday)
        is_weekend = 0
        if d.weekday() > 4:
            is_weekend = 1
        weekend.append(is_weekend)
        minute.append((d.hour * 60) + d.minute)

    df['Week'] = week
    df['Month'] = month
    df['Day_of_year'] = doy
    df['Day'] = dow
    df['Minutes'] = minute
    df['Weekend'] = weekend

    df.to_csv(filename)


if __name__ == '__main__':

    for i in ['01']: #, '02', '03', '04', '05', '06', '07', '08']:

        filename = 'Datasets/' + i + '/' + i + 'final.csv'
        #tmp_date(filename)
        df = pd.read_csv(filename)
        cons = df['Consumption(Wh)'].values
        slp = df['slp_forecast'].values

        print(mean_absolute_percentage_error(df['Consumption(Wh)'].values, slp))
        """
        cons = df['Consumption(Wh)'].values
        cons = cons[:35040]
        total_cons = np.sum(cons)
        slp = df['slp'].values
        slp_forecast = slp*total_cons
        df['slp_forecast']=slp_forecast
        df.to_csv(filename)
        """


    """
    dt = df['Datetime'][:64952]
    dates = [datetime.strptime(d, "%d/%m/%Y %H:%M") for d in dt]

    curr_date = dates[-1]
    print(curr_date)

    for i in range(len(df)-len(dates)):
        next = curr_date+timedelta(minutes=15)
        curr_date = next
        dates.append(next)
    df['Datetime'] = dates
    print(df['Datetime'])
    #df.to_csv('Datasets/anomaly_test/anomaly_test2.csv')

    
    #print(cons.var())
    #tmp_cons_calcu(filename)
    #tmp_time_features(filename)
    #tmp_date(filename)
    #mean_cons_by_hour2(filename)

    df = pd.read_csv(filename)
    dt = df['Datetime']
    dupp = dt.duplicated()
    for i in range(len(dupp)):
        if dupp[i]:
            print(i)

 
  
    df = pd.read_csv(filename)
    hour = df['Hour']
    for i in range(len(hour)):
        try:
            if hour[i][3:6] not in (['00', '07', '30', '45']):
                print(hour[i][3:6])
                print(i)
        except:
            print(i)
    #print(np.mean(df['Consumption(Wh)']))
    """
