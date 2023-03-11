import helper
from RandomForest import random_forest_model

if __name__ == '__main__':

    variables = ["Minutes", "Day", "Week", "Weekend", "Month", "Temperature", "Humidity", "Pressure",
                 "Wind speed", "Wind direction", "Snowfall", "Snow depth", "Irradiation", "Rainfall"]
    res = helper.select_best_features('test.csv', random_forest_model, variables=variables)

    print("Final result :")
    print(res[0])
    print(res[1])
 
    with open('Select_best_var_RF_10.txt', 'w') as f:
        for i in range(len(res[0])):
            f.write(res[0][i] + ': ' + str(res[1][i]) + '\n')
