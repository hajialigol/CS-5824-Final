import pandas as pd
import ssl

# runner
if __name__ == "__main__":

    # get path to save df
    save_path = fr"C:\Users\Haji\Documents\CS\5824\CS-5824-Final\data\crypto.csv"

    # get list of crypto coins to look at
    crypto_list = ['BTC', 'ETH', 'LTC', 'NEO', 'BNB', 'XRP',
                   'LINK', 'EOS', 'TRX', 'ETC', 'XLM', 'ZEC']

    # use unverified ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    # get bitcoin data
    btc_url = f"https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_d.csv"
    df = pd.read_csv(btc_url, delimiter=',', skiprows=[0])

    # only keep relevant columns
    df = df[['date', 'open', 'high', 'low', 'close']]

    # iterate over coin
    for i in range(len(crypto_list) - 1):

        # get bitcoin data
        url = f"https://www.cryptodatadownload.com/cdd/Binance_{crypto_list[i+1]}USDT_d.csv"
        crypto_df = pd.read_csv(url, delimiter=',', skiprows=[0])

        # only keep relevant columns
        crypto_df = crypto_df[['date', 'open', 'high', 'low', 'close']]

        # make an inner join on the data frames based on dates
        df = df.merge(crypto_df, on='date', how='inner', suffixes=(f'_{crypto_list[i]}',
                                                                   f'_{crypto_list[i+1]}'))

    # reorder the dataframe by oldest prices
    df = df.sort_values('date')

    # save dataframe
    df.to_csv(save_path)
