import numpy as np
import pandas as pd

from trading import Trading
from dqn import DQNAgent

EPISODES = 10

class BrokerageTrading(Trading):
    """
    Observation:
        Type: Box(Window+2)
        Num             Observation             Min         Max
        0               Remaining Trading days  0           Span or Max
        1               Cash                    0           Inf
        2               Number of Stock Owned   0           Inf
        3..(Window+3)   Window-day Prices       0           Inf

    Actions:
        Type: Discrete(3)
        Num Action
        0   Buy
        1   Hold
        2   Sell

    Reward:
        Reward is _ for every step taken, including the termination step
            - Total Cash
            - Profit Gain

    Starting State:
        Initial prices in the first Window-day window
        Number of Stock Owned   0
        Cash                    10000

    Episode Termination:
        The trading interval is finished (= Span days)
        Total cash <= 0, and Number of Stock Owned <= 0
        Episode length is greater than 200
    """


    def history(self):
        return self.stock['Close'][self.today()-self.window:self.today()]
        # np.diff(self.stock['Close'][self.today()-self.window-1:self.today()])/ \
                        # self.stock['Close'][self.today()-self.window:self.today()]

    def observations(self, cash, nown, p):
        flat_rate = 0.0 #5 dollars per transaction
        percent_rate = 0.0 #0.5% fees

        n = np.floor(cash/p)

        fee_sell = int(nown != 0)*flat_rate + p*nown*percent_rate
        fee_buy = int(n != 0)*flat_rate + p*n*percent_rate
        while cash - p*n - fee_buy < 0:
            n -= 1
            fee_buy = flat_rate + p*n*percent_rate

        return np.array([
            [cash-p*n-fee_buy,      nown+n], # BUY
            [cash,                  nown],   # HOLD
            [cash+p*nown-fee_sell,  0],      # SELL
        ])

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        state = self.state
        asset_before = self.eval(state[1], state[2])

        done = (self.i == 0)
        if not done:
            holdings = self.observations(state[1], state[2], self.price())[action]
            self.i -= 1
            self.state = np.hstack([[self.i, holdings[0], holdings[1]], self.history()])
            reward = self.eval(*holdings) - asset_before
            if reward == 0: reward = -2
            print('start', self.start, self.i, 'previous', (state[1], state[2]), 'current', holdings, 'action', action, 'reward', reward)

        else:
            holdings = self.observations(state[1], state[2], self.price())[2]
            self.state = np.hstack([[self.i, holdings[0], holdings[1]], self.history()])
            print('start', self.start, self.i, 'previous', (state[1], state[2]), 'current', holdings)
            reward = 0.0

        return np.array(self.state), reward, done, {}


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--list',
    #                 const='all',
    #                 nargs='?',
    #                 choices=['train, test'])

    #parser.add_argument("test")
    # args = parser.parse_args()
    # print(args)


    for stock_name in ['linear']:
        print('Start training for stock ' + stock_name + '...\n')
        env = Trading(stock_name)
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

# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")

# # later...

# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
