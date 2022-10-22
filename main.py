import pandas as pd

def data_preprocessing():
    df = pd.read_csv("../donneeconso10.csv")[["Date", "Heure", "Index(Wh)"]]
    one_year = df.loc[34:35296]
    one_year_df = pd.DataFrame(one_year, columns=df.columns)
    one_year_df.reset_index(drop=True, inplace=True)
    one_year_df.to_csv("one_year10.csv")


if __name__ == '__main__':
    data_preprocessing()