import pandas as pd
from src.dataset import CryptoDataset

# runner
if __name__ == "__main__":

    # get path to data
    data_path = r'C:\Users\Haji\Documents\CS\5824\CS-5824-Final\data\crypto.csv'

    # read the csv file
    df = pd.read_csv(data_path)

    # create a crypto dataset object
    crypto_ds = CryptoDataset(df)