import numpy as np
import pandas as pd

from trading import Trading

from gym import Env, spaces
from dqn import DQNAgent

EPISODES = 1000

class Longshort(Trading):
    def __init__(self, symbol, cash=10000, window=30, span=100, start=None):
        super().__init__(symbol, cash, window, span, start)

        high = np.array([np.finfo(np.float32).max]*(self.window+3)) # TODO limit num remaining trading days
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

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

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        state = self.state
        asset_before = self.eval(state[1], state[2])

        done = (self.i == 0)
        if not done:
            # print(self.observations(state[1], state[2], self.price()))
            holdings = self.observations(state[1], state[2], self.price())[action]
            self.i -= 1
            self.state = np.hstack([[self.i, holdings[0], holdings[1]], self.history()])
            reward = self.eval(*holdings) - asset_before
            # print('start', self.start, self.i, 'previous', (state[1], state[2]), 'current', holdings, 'action', action, 'reward', reward)

        else:
            holdings = [state[1] + state[2] * self.price(), 0] # sell all
            self.state = np.hstack([[self.i, holdings[0], holdings[1]], self.history()])
            print('start', self.start, self.i, 'previous', (state[1], state[2]), 'current', holdings)
            reward = 0.0

        return np.array(self.state), reward, done, {}


if __name__ == '__main__':
    for stock_name in ['linear']:
        print('Start training for stock ' + stock_name + '...\n')
        env = Longshort(stock_name, span=100)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        agent = DQNAgent(state_size, action_size)
        #agent.save("./save/blank_trading.h5")
        load_string = './save/' + stock_name + '_weights_with_fees.h5'
        # agent.load(load_string)
        done = False
        batch_size = 64

        for e in range(EPISODES):
            print('training episode:', e, '\n')
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            for time in range(500):
                # env.render()
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                # reward = reward if not done else -10
                next_state = np.reshape(next_state, [1, state_size])
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                # print(action, reward)
                if done:
                    print("episode: {}/{}, score: {}, e: {:.5}"
                          .format(e, EPISODES, time, agent.epsilon))
                    break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            if e % 10 == 0:
                save_string = './save/' + stock_name + '_weights_with_fees.h5'
                agent.save(save_string)

