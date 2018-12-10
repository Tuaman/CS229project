import os

import numpy as np
import pandas as pd

from dqn import DQNAgent
from grapher import Grapher
from gym import spaces
from trading import Trading


EPISODES = 1000


class Longshort(Trading):
    def __init__(self, symbol, cash=10000, window=30, span=100, start=None):
        super().__init__(symbol, cash, window, span, start)

        self.action_labels = ['LONG','SHORT']
        self.action_space = spaces.Discrete(len(self.action_labels))
        self.done_action = self.action_labels.index('SHORT')

        high = np.array([np.finfo(np.float32).max]*(self.window)) # TODO limit num remaining trading days
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def get_state(self, i=None, cash=None, nown=None):
        i    = i    or self.i
        cash = cash or self.holdings[0]
        nown = nown or self.holdings[1]

        # state = np.hstack([[i, cash, nown], self.history()])
        state = self.history()
        assert self.observation_space.shape[0] == state.shape[0], "%d (expected %d) invalid state size"%(state.shape[0], self.observation_space.shape[0])

        return state

    def observations(self, cash, nown, p):
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


if __name__ == '__main__':
    for stock_name in ['sine_50days']:
        print('Start training for stock ' + stock_name + '...\n')

        env = Longshort(stock_name, window=2, span=100)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        agent = DQNAgent(state_size, action_size)
        #agent.save("./save/blank_trading.h5")
        # load_string = './save/' + stock_name + '_weights_without_fees.h5'
        #agent.load(load_string)
        done = False
        batch_size = 64

        title = env.symbol.upper()+' '+os.path.basename(__file__).split('.')[0]
        grapher = Grapher(title, action_labels=env.action_labels)

        for e in range(EPISODES):
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            for time in range(500):
                cash, nown, price = env.holdings[0], env.holdings[1], env.state[-1]
                # env.render()
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                # reward = reward if not done else -10
                next_state = np.reshape(next_state, [1, state_size])
                agent.remember(state, action, reward, next_state, done)
                if e % 50 == 0:
                    grapher.add(cash, nown, price, action, reward)

                agent.train(state, action, reward, next_state, done)
                state = next_state
                if done:
                    print('start', env.start, 'previous', (cash, nown), 'current', tuple(env.holdings))
                    print("episode: {}/{}, score: {}, e: {:.5}"
                          .format(e, EPISODES, time, agent.epsilon))
                    if e % 50 == 0:
                        grapher.show(ep=e, t=time, e=agent.epsilon)
                        grapher.reset()
                    break

                #if len(agent.memory) > batch_size:
            if e % 100 == 0:
                save_string = './save/' + stock_name + '_weights_without_fees.h5'
                agent.save(save_string)
