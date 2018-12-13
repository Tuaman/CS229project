import numpy as np
import pandas as pd

class Trader:
    def __init__(self, mode='buy-sell'):
        self.mode = mode

    BUY, LONG   = 0, 0
    HOLD, SHORT = 1, 1
    SELL        = 2
    def to_strategy(self, prev_price, prices):
        strategy = np.zeros(prices.shape[0], dtype=int)
        if self.mode == 'buy-sell':
            strategy[prev_price < prices] = Trader.BUY
            strategy[prev_price == prices] = Trader.HOLD
            strategy[prev_price > prices] = Trader.SELL
        else:
            strategy[prev_price <= prices] = Trader.LONG
            strategy[prev_price > prices] = Trader.SHORT

        return strategy

    def long_short(self, cash, nown, p):
        flat_rate = 0 #5 dollars per transaction
        percent_rate = 0 # 0.5% fees
        short_fee = 0  #per-day fees

        cash_tot = cash + nown * p
        n_short = np.floor(cash_tot/p)
        n_long = n_short

        # fee_short = int(nown == -n_short) * (flat_rate + abs(n_short - nown) * percent_rate)
        # fee_long = int(nown == n_long) * (flat_rate + abs(n_long - nown) * percent_rate)
        # while cash_tot-p*n_long-fee_long < 0:
        #     n_long -= 1
        #     fee_long = int(nown == n_long) * (flat_rate + abs(n_long - nown) * percent_rate)

        fee_long = 0
        fee_short = 0
        return np.array([
            [cash_tot-p*n_long-fee_long,     n_long], # LONG
            [cash_tot+p*n_short-fee_short,  -n_short], # SHORT
        ])

    def buy_sell(self, cash, nown, p):
        n = np.floor(cash/p)
        return np.array([
            [cash-p*n,      nown+n], # BUY
            [cash,          nown],   # HOLD
            [cash+p*nown, 0],        # SELL
        ])

    def run(self, strategy, prices, cash=10000, grapher=None):
        observations = self.buy_sell if self.mode == 'buy-sell' else self.long_short

        stock_owned = 0
        for i in range(strategy.shape[0]-1):
            prev_cash, prev_nown = cash, stock_owned
            cash, stock_owned = observations(cash, stock_owned, prices[i])[strategy[i]]

            if grapher:
                reward = self.eval(cash, stock_owned, prices[i+1]) - self.eval(prev_cash, prev_nown, prices[i])
                grapher.add(prev_cash, prev_nown, prices[i], strategy[i], reward)

        return self.eval(cash, stock_owned, prices[-1])

    def eval(self, cash, stock_owned, price):
        return cash + stock_owned * price

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

