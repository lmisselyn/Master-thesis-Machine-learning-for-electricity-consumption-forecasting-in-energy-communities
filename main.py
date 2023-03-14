import helper
from helper import select_best_features, one_week_test
from Models.mlp_regression import mlp_model
from Models.RandomForest import random_forest_model
from Models.XGB import XGB_regressor_model
import sys

models = {'Random_forest' : random_forest_model, "MLP" : mlp_model, "XGB" : XGB_regressor_model}
variables = ["Minutes", "Day", "Week", "Weekend", "Month", "Temperature", "Humidity", "Pressure",
             "Wind speed", "Wind direction", "Snowfall", "Snow depth", "Irradiation", "Rainfall"]

def test_file(filename):
    try:
        f = open(filename, 'r')
        f.close()
    except FileNotFoundError:
        sys.exit("second argument must be path to a .csv file")

def test_model_name(model):
    if args[2] not in models.keys():
        sys.exit("Wrong model name, available models are : " + str(models.keys()))


if __name__ == '__main__':

    args = sys.argv[1:]

    if args[0] == "one_week_test":
        test_file(args[1])
        test_model_name(args[2])
        one_week_test(args[1], args[2], helper.get_features(args[1], args[2]))

    elif args[0] == "select_best_features":
        test_file(args[1])
        test_model_name(args[2])
        select_best_features(args[1], models[args[2]], variables=variables)

    else:
        print("First argument must be either \'one_week_test\' or \'select_best_features\'")

