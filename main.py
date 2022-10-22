import pandas

def data_preprocessing():
    df = pandas.read_csv("../donneeconso10.csv")
    print(df["Date"])

if __name__ == '__main__':
    data_preprocessing()