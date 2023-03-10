import helper
from RandomForest import random_forest_model

if __name__ == '__main__':

    variables = ["Minutes", "Day", "Week", "Weekend", "Month", "Temperature", "Humidity", "Pressure",
                 "Wind speed", "Wind direction", "Snowfall", "Snow depth", "Irradiation", "Rainfall"]
    selected, acc = helper.select_best_features('one_year_10.csv', random_forest_model, variables=variables)
 
    with open('Select_best_var_RF_10.txt', 'w') as f:
        for i in range(len(selected)):
            f.write(selected[i] + ': ' + str(acc[i]) + '\n')
    