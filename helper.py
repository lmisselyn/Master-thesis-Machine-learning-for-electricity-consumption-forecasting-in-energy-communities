import pandas as pd


def make_sets(filename):
    """
    parameters ;
        filename : name of the csv file
    return ;
        training set, validation set, test set
    """
    df = pd.read_csv(filename)
    data = [[], [], []]
    i = 0
    for j in range(len(df)):
        data[i].append(df.loc[j])
        i += 1
        if i > 2:
            i = 0
    train_df = pd.DataFrame(data[0], columns=df.columns)
    validation_df = pd.DataFrame(data[1], columns=df.columns)
    test_df = pd.DataFrame(data[2], columns=df.columns)
    return train_df, validation_df, test_df
