import numpy as np
import pandas as pd
import gym

from Model import Model
from Trader import Trader

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

class supervised_learning(Model):

    def __init__(self, symbol, cash=10000, window=30, span=100, start=None):
        self.stock_name = symbol
        self.memory = deque(maxlen=2000)
        self.cash = cash
        self.holdings = [cash, 0]
        self.span = span
        self.window = window
        self.start = start
        self.learning_rate = 0.005
        self.stock = pd.read_csv('data/'+symbol+'.us.csv')
        self.state = None
        self.model = self._build_model()

    def price(self):
        return self.stock['Close'][self.today]

    def history(self):
        return self.stock['Close'][self.today-self.window:self.today].values

    def net_worth(self):
        return self.holdings[0] + self.holdings[1] * self.price()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.window, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        # plot_model(model, to_file='model.png')
        return model

    def observations(self, cash, nown, p):
        flat_rate = 0 #5 dollars per transaction
        percent_rate = 0# 0.5% fees
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

    def train(self, epoch):
        print('training on', self.stock_name)
        for i in range(epoch):
            print('Training {}/{}'.format(i, epoch))
            self.reset()
            for _ in range(self.end - self.start + 1):
                self.today += 1
                target = self.price().reshape(1, 1)
                state = np.reshape(self.state, [1, len(self.state)])
                self.model.fit(state, target , epochs=1, verbose=0)
                self.state = self.history()
                

    def trade(self):
        self.reset()
        self.today = np.random.randint(self.window, self.stock.shape[0]-self.span)
        for i in range(self.span):
            self.state = self.history()
            state = np.reshape(self.state, [1, len(self.state)])
            predict_price = self.model.predict(state)
            if predict_price >= self.price():
                self.holdings = self.observations(*self.holdings, self.price())[0]
                action = 'long'
            else:
                self.holdings = self.observations(*self.holdings, self.price())[1]
                action = 'short'
            print('today_price', self.price(), 'predict_price', predict_price, \
                'holding', self.holdings, 'action:', action, 'net_worth:', self.net_worth())
            self.today += 1
        return self.net_worth()

    def reset(self):
        self.start = self.window
        self.today = self.start
        self.end = self.stock.shape[0]-self.span
        self.state = self.history()
        return self.state

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    stock_name = 'sine_50days'
    trader = supervised_learning(stock_name)
    save_string = './save/supervised' + stock_name +'.h5'
    #trader.load(save_string)
    trader.train(1)
    final_net_worth = trader.trade()
    #trader.save(save_string)
    print("final_net_worth = ", final_net_worth)




