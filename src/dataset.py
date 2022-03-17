import numpy as np
import torch

# dataset that represents the input to the model, X_{t}
class CryptoDataset():

    # initializer
    def __init__(self, df, f=3, n=50, m=12):
        # dataframe consisting of crypto currency data
        self.df = df
        # f is the feature number, which is 3 in the paper.
        self.f = f
        # n represents the number of time periods before
        # a given time period. In the paper, this is set at 50
        self.n = n
        # m represents the number of cryptocurrencies
        self.m = m
        # list that contains all the symbols for the cryptocurrencies
        # that will be inputted in the model
        self.crypto_list = ['BTC', 'ETH', 'LTC', 'NEO', 'BNB', 'XRP',
                            'LINK', 'EOS', 'TRX', 'ETC', 'XLM', 'ZEC']
        # get the columns names that contain the closing price
        self.closing_names = [x for x in self.df.columns if 'close' in x]
        # get the columns names that contain the high price
        self.high_names = [x for x in self.df.columns if 'high' in x]
        # get the columns names that contain the high price
        self.low_names = [x for x in self.df.columns if 'low' in x]
        # create empty tensor that will store results
        self.Vt_closing = torch.zeros((self.n, self.m))
        # create empty tensor that will store results
        self.Vt_high = torch.zeros((self.n, self.m))
        # create empty tensor that will store results
        self.Vt_low = torch.zeros((self.n, self.m))

    # compute V_t (closing price)
    def compute_V_t_closing(self, t):

        # get the starting index for the corresponding row in the dataframe
        # note: it's guaranteed that this index will be >= 0
        # t-n+1: t -> t-n: t because python is 0 index based
        idx = t - self.n

        # get the section of the dataframe that will be iterated over
        local_df = self.df[idx: t] # "+1" because python doesn't get element at t

        # get values for v_t (needed for element-wise division
        v_t = local_df.loc[t-1, self.closing_names].values

        # index for adding price relative (pr) vector to the
        # Vt tensor
        pr_idx = 0

        # iterate over the local dataframe
        for i, row in local_df.iterrows():

            # get the values for v_i
            v_i = row[self.closing_names].values

            # compute y_i (element-wise multiplication)
            y_i = torch.from_numpy(
                    np.divide(v_i, v_t).astype('float')
            )

            # make the first element 1 because the first element
            # is the quoted currency (called the "cash" in the paper)
            # is considered to be 1 for all t (v_{0, t} = 1))
            y_i[0] = 1

            # add to the tensor
            self.Vt_closing[pr_idx] = y_i

            # increment counter
            pr_idx += 1


    # compute V_t (high price)
    def compute_V_t_high(self, t):

        # get the starting index for the corresponding row in the dataframe
        # note: it's guaranteed that this index will be >= 0
        # t-n+1: t -> t-n: t because python is 0 index based
        idx = t - self.n

        # get the section of the dataframe that will be iterated over
        local_df = self.df[idx: t] # "+1" because python doesn't get element at t

        # get values for v_t (needed for element-wise division
        v_t = local_df.loc[t-1, self.closing_names].values

        # index for adding price relative (pr) vector to the
        # Vt tensor
        pr_idx = 0

        # iterate over the local dataframe
        for i, row in local_df.iterrows():

            # get the values for v_i
            v_i = row[self.high_names].values

            # compute y_i (element-wise multiplication)
            y_i = torch.from_numpy(
                    np.divide(v_i, v_t).astype('float')
            )

            # make the first element 1 because the first element
            # is the quoted currency (called the "cash" in the paper)
            # is considered to be 1 for all t (v_{0, t} = 1))
            y_i[0] = 1

            # add to the tensor
            self.Vt_high[pr_idx] = y_i

            # increment counter
            pr_idx += 1


    # compute V_t (low price)
    def compute_V_t_low(self, t):

        # get the starting index for the corresponding row in the dataframe
        # note: it's guaranteed that this index will be >= 0
        # t-n+1: t -> t-n: t because python is 0 index based
        idx = t - self.n

        # get the section of the dataframe that will be iterated over
        local_df = self.df[idx: t] # "+1" because python doesn't get element at t

        # get values for v_t (needed for element-wise division
        v_t = local_df.loc[t-1, self.closing_names].values

        # index for adding price relative (pr) vector to the
        # Vt tensor
        pr_idx = 0

        # iterate over the local dataframe
        for i, row in local_df.iterrows():

            # get the values for v_i
            v_i = row[self.low_names].values

            # compute y_i (element-wise multiplication)
            y_i = torch.from_numpy(
                    np.divide(v_i, v_t).astype('float')
            )

            # make the first element 1 because the first element
            # is the quoted currency (called the "cash" in the paper)
            # is considered to be 1 for all t (v_{0, t} = 1))
            y_i[0] = 1

            # add to the tensor
            self.Vt_low[pr_idx] = y_i

            # increment counter
            pr_idx += 1
