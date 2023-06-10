from datetime import datetime, timedelta

import requests
import pandas as pd


#url = ("https://archive-api.open-meteo.com/v1/archive?latitude=50.8706&longitude=4.3513&start_date=2020-01-01&end_date=2023-01-01&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,apparent_temperature,pressure_msl,surface_pressure,snowfall,weathercode,cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high,shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,windspeed_10m,winddirection_10m")

url = ("https://archive-api.open-meteo.com/v1/archive?latitude=50.8706&longitude=4.3513&start_date=2020-01-01&end_date=2023-01-01&hourly=precipitation")
df = pd.read_json(url)


tr = df.transpose()

data = {}
for c in tr.columns:
    data[c] = tr.at['hourly', c]

new_df = pd.DataFrame(data)
Date=[]
for t in new_df['time']:
    Date.append(datetime.strptime(t, "%Y-%m-%dT%H:%M"))
new_df['Date'] = Date

df = new_df
print(df)
df.drop(columns='time', inplace=True)
data = []

for i in range(len(df)-1):
    new_cols = [[],[],[]]
    before = df.iloc[i]
    after = df.iloc[i+1]
    data.append(before.values)
    for c in df.columns:
        if c == 'Date':
            for j in range(3):

                new_cols[j].append(datetime.fromisoformat(str(before['Date']))+timedelta(minutes=15*(j+1)))
        else:
            diff = (after[c]-before[c])/4
            for j in range(3):
                new_cols[j].append(before[c]+(diff*(j+1)))

    for j in range(3):
        data.append(new_cols[j])

print(data[:20])
meteo = pd.DataFrame(data, columns=df.columns)
print(meteo.iloc[:100])

meteo.to_csv('rainfall.csv')

