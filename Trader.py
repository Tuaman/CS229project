import numpy as np
import pandas as pd

class Trader:
    BUY = 1
    SELL = -1

    def __init__(self, stock):
        self.stock = stock

    def run(self, strategy, cash=10000, span=None):
        if span is None:
            span = len(strategy)

        stock_owned = 0
        for i in range(-span, 0):
            price = self.stock['Close'][self.stock.shape[0]+i]
            if strategy[i] == Trader.BUY:
                num_stock = np.floor(cash/price)
                cash -= num_stock * price
                stock_owned += num_stock
            elif strategy[i] == Trader.SELL:
                cash += stock_owned * price
                stock_owned = 0
        # print(cash, stock_owned, self.stock['Close'][self.stock.shape[0]-1], self.stock['Close'][self.stock.shape[0]+i], i, self.stock.shape[0]+i, self.net_total(cash, stock_owned))
        return self.net_total(cash, stock_owned)

    def net_total(self, cash, stock_owned):
        return cash + stock_owned * self.stock['Close'][self.stock.shape[0]-1]

    def buy_once_and_hold(self, buy_at=0):
        strategy = np.zeros(self.stock.shape[0])
        strategy[buy_at], strategy[-1] = Trader.BUY, Trader.SELL
        return strategy


if __name__ == "__main__":
    stock_name = 'amzn'
    stock = pd.read_csv('data/'+stock_name+'.us.csv')

    td = Trader(stock)
    # Buy at the very beginning and sell all by the end
    strategy = td.buy_once_and_hold()
    total = td.run(strategy)
    print(stock_name+': '+str(total))

