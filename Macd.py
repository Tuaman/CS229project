import numpy as np
import pandas as pd

from Model import Model
from Trader import Trader

class Macd(Model):

    def macd(self, stock):
        ema12 = stock['Close'].ewm(span=12, min_periods=0, adjust=True, ignore_na=False).mean()
        ema26 = stock['Close'].ewm(span=26, min_periods=0, adjust=True, ignore_na=False).mean()
        macd = ema12 - ema26

        signal = macd.ewm(span=9, min_periods=0, adjust=True, ignore_na=False).mean()
        return macd, signal

    def predict(self, stock):
        macd, signal = self.macd(stock)

        macd_above = macd >= signal
        prev_macd_above = macd_above[:-1].values
        macd_above = macd_above[1:].values

        y = np.zeros(macd_above.shape)
        y[macd_above & ~prev_macd_above] = Trader.BUY
        y[~macd_above & prev_macd_above] = Trader.SELL
        return y


if __name__ == "__main__":
    stock_name = 'amzn'
    stock = pd.read_csv('data/'+stock_name+'.us.csv')

    mc = Macd()
    strategy = mc.predict(stock)
    # print(sum(strategy))
