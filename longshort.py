import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dqn import DQNAgent
from grapher import Grapher
from gym import spaces
from trading import Trading


EPISODES = 2000


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

    def step(self, action):
        # steps:
        # 1. Calculate asset before trading
        # 2. Take action by reallocating the asset using the price of today
        # 3. Move on to the next day, then calculate reward
        # 4. Get new state, then return state, reward, done
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        state = self.state
        holdings = self.holdings
        asset_before = self.eval(*self.holdings)
        price = self.price()

        done = (self.i == 1)
        self.holdings = self.observations(*self.holdings, self.price())[action]
        self.i -= 1
        self.state = self.get_state()
        reward = self.eval(*self.holdings) - asset_before
        if self.i % 20 == 0:
            print(self.i, 'price', price, 'previous', holdings, 'current', self.holdings, 'action', action, 'reward', reward)
        
        # else:
        #     self.holdings = [self.holdings[0] + self.holdings[1] * self.price(), 0] # sell all
        #     self.state = self.get_state()
        #     reward = 0.0
        return np.array(self.state), reward, done, {}


if __name__ == '__main__':
    for stock_name in ['sine_50days']:
        print('Start training for stock ' + stock_name + '...\n')

        env = Longshort(stock_name, window=30, span=100)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        agent = DQNAgent(state_size, action_size)
        save_string = './save/' + stock_name + '_weights_without_fees_test.h5'
        #agent.load(save_string)
        done = False
        batch_size = 64

        title = env.symbol.upper()+' MDP '+os.path.basename(__file__).split('.')[0]
        grapher = Grapher(title, action_labels=env.action_labels)

        with open('./save/losses_'+stock_name+'.txt', 'w') as f:
            # plt.axis([0, 10, 0, 1])
            for e in range(EPISODES):
                state = env.reset()
                state = np.reshape(state, [1, state_size])
                for time in range(500):
                    cash, nown, price = env.holdings[0], env.holdings[1], env.state[-1]
                    # env.render()
                    action = agent.act(state, time)
                    next_state, reward, done, _ = env.step(action)
                    next_state = np.reshape(next_state, [1, state_size])
                    agent.remember(state, action, reward, next_state, done)
                    agent.train(state, action, reward, next_state, done)
                    if e % 25 == 0:
                        #cash, nown, price = state[0, 1], state[0, 2], state[0, -1]
                        # cash, nown, price = *env.holdings, state[0,-1]
                        grapher.add(cash, nown, price, action, reward, loss=agent.loss)
                        #print(action, reward)

                    state = next_state
                    if done:
                        print('start', env.start, 'previous', (cash, nown), 'current', tuple(env.holdings))
                        print("episode: {}/{}, score: {}, e: {:.5}"
                              .format(e, EPISODES, time, agent.epsilon))
                        print('average_loss =', agent.loss/env.init['span'])
                        f.write(str(agent.loss)+'\n')
                        f.flush()
                        agent.loss = 0
                        if e % 25 == 0:
                            grapher.show(ep=e, t=time, e=agent.epsilon)
                            grapher.reset()
                            agent.save(save_string)
                        break
                    # if len(agent.memory) > batch_size:
                    #     agent.replay(batch_size)


        #     plt.plot(losses)
        #     plt.pause(0.05)
        # plt.show()


