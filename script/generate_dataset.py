import pandas as pd
def dataset(X, Y, train_size, test_size, dataset_name, output_path):

    with open(f'{output_path}/{dataset_name}_{train_size}_{test_size}.txt', 'w') as f:
        f.write(f"{train_size} {X.shape[1]}\n")
        X.iloc[:train_size, :].to_csv(f, index=False, header=False, sep=' ')
        f.write(f"{train_size} 1\n")
        Y.iloc[:train_size, :].to_csv(f, index=False, header=False, sep=' ')

        f.write(f"{test_size} {X.shape[1]}\n")
        X.iloc[train_size:train_size+test_size, :].to_csv(f, index=False, header=False, sep=' ')
        f.write(f"{test_size} 1\n")
        Y.iloc[train_size:train_size+test_size, :].to_csv(f, index=False, header=False, sep=' ')

if __name__ == "__main__":
    # PRSA dataset
    # filename = 'data/PRSA/PRSA_data_2010.1.1-2014.12.31.csv'
    # df = pd.read_csv(filename)
    # X_cols = ['year',  'month', 'day',  'hour', 'DEWP','TEMP','PRES', 'Iws','Is','Ir']
    # Y_cols = ['pm2.5']
    # dataset(X, Y, 2000, 200, 'PRSA', 'data/PRSA')

    # wine dataset
    filename = 'data/wine/winequality-red.csv'
    df = pd.read_csv(filename, sep=';')
    X_cols = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]
    Y_cols = ["quality"]

    df = df.dropna(axis=0)
    X = df[X_cols]
    Y = df[Y_cols]
    dataset(X, Y, 1300, 299, 'redwine', 'data/wine')