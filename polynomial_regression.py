import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def regress_visu(filename, variables):
    df = pd.read_csv(filename)
    for v in variables:
        x = df[v]
        y = df["Consumption(Wh)"]
        model = np.poly1d(np.polyfit(x, y, 3))
        fig, ax = plt.subplots()
        ax.plot(x, y, linewidth=0, marker='s', label='Data points')
        ax.plot(x, model(x), label='polynomial_regress')
        ax.set_xlabel(v)
        ax.set_ylabel("Consumption(Wh)")
        ax.legend(facecolor='white')
        plt.show()


if __name__ == '__main__':
    variables = ["Day", "Week", "Weekend", "Month", "Temperature", "Humidity", "Pressure",
                "Wind speed", "Wind direction", "Snowfall", "Snow depth", "Irradiation", "Rainfall"]
    regress_visu('one_year_10.csv', variables)