import os
import numpy as np
import pandas as pd
import gym

from Model import Model
from Trader import Trader
from processor import Processor

from grapher import Grapher
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

from sklearn.metrics import accuracy_score, balanced_accuracy_score, average_precision_score, f1_score


class NN(Model):
    def __init__(self, input_dim, learning_rate=0.001):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.input_dim, activation='relu'))
        model.add(Dense(24, activation='relu'))
        # model.add(Dense(12, activation='relu'))
        # model.add(Dense(12, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        # plot_model(model, to_file='model.png')
        return model

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class NN12(NN):
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(12, input_dim=self.input_dim, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        # plot_model(model, to_file='model.png')
        return model

class NN24(NN):
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.input_dim, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        # plot_model(model, to_file='model.png')
        return model


# if __name__ == "__main__":
#     stock_name = 'sine_50days'
#     trader = NN(stock_name)
#     save_string = './save/supervised' + stock_name +'.h5'
#     trader.load(save_string)
#     #trader.train(50)
#     final_net_worth = trader.trade()
#     trader.save(save_string)
#     print("final_net_worth = ", final_net_worth)

EPISODES = 2000

if __name__ == "__main__":
    stock_name = 'amzn'
    stock = pd.read_csv('data/'+stock_name+'.us.csv')

    span = 300
    window = 100
    proc = Processor(stock, window=window)
    x_test, y_test = proc.run(span=span, start=stock.shape[0]-3*span-1)

    input_dim = proc.window
    models = {'NN24x6': NN24(input_dim)}
    # modes = {'buy-sell': ['BUY', 'HOLD', 'SELL'], 'long-short': ['LONG', 'SHORT']}
    modes = {'long-short': ['LONG', 'SHORT']}

    for name, model in models.items():
        title = stock_name.upper()+' '+name
        grapher = Grapher(title)
        with open(title.lower().replace(' ', '_')+'.txt', 'w') as f:
            for e in range(EPISODES+1):
                x_train, y_train = proc.run(span=span)

                model.model.fit(x_train, y_train)
                y_pred = model.model.predict(x_test)

                prev_prices, prices, pred = x_test[:, -1], y_test, y_pred.reshape(y_pred.shape[0])
                if e % 100 == 0:
                    for mode, action_labels in modes.items():
                        trader = Trader(mode)
                        strategy = trader.to_strategy(prev_prices, pred)
                        optimal = trader.to_strategy(prev_prices, prices)
                        trader.run(strategy, prices, grapher=grapher)

                        # f.write('{} accuracy_score={} balanced_accuracy_score={} average_precision_score={} f1_score={}\n'.format(
                        #     e,
                        #     accuracy_score(strategy, optimal),
                        #     balanced_accuracy_score(strategy, optimal),
                        #     average_precision_score(strategy, optimal),
                        #     f1_score(strategy, optimal)))

                        grapher.action_labels = action_labels
                        grapher.pred = pred[:-1]

                        grapher.show(action_labels=action_labels, ep=e, span=span, w=window, mode=trader.mode)
                        grapher.reset()
