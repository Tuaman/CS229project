import numpy as np
import pandas as pd

from gym import Env, spaces
from dqn import DQNAgent

EPISODES = 10

class Trading(Env):
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

    def __init__(self, symbol, cash=10000, window=30, span=300, start=None):
        self.stock = pd.read_csv('data/'+symbol+'.us.csv')

        self.span = span
        self.window = window
        self.cash = cash

        low = np.zeros(self.window+3)
        high = np.array([np.finfo(np.float32).max]*(self.window+3)) # TODO limit num remaining trading days

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.state = None

    def _history(self):
        return self.stock['Close'][self.today()-self.window:self.today()]

    def today(self):
        return self.start+self.span-self.i

    def _observations(self, cash, nown, p):
        n = np.floor(cash/p)
        return np.array([
            [cash-p*n,      nown+n], # BUY
            [cash,          nown],   # HOLD
            [cash+p*nown, 0],        # SELL
        ])

    def price(self):
        return self.stock['Close'][self.today()]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        state = self.state
        asset_before = self.eval(state[1], state[2])

        done = (self.i == 0)
        if not done:
            holdings = self._observations(state[1], state[2], self.price())[action]
            self.i -= 1
            self.state = np.hstack([[self.i, holdings[0], holdings[1]], self._history()])
            reward = self.eval(*holdings) - asset_before
        else:
            holdings = self._observations(state[1], state[2], self.price())[2]
            self.state = np.hstack([[self.i, holdings[0], holdings[1]], self._history()])
            print('start', self.start, self.i, 'previous', (state[1], state[2]), 'current', holdings)
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def eval(self, c, n):
        return c + n*self.price()

    def reset(self):
        self.i = self.span
        self.start = np.random.randint(self.window, self.stock.shape[0]-self.span)
        self.state = np.hstack([np.array([self.i, self.cash, 0]), self._history()])
        return self.state


if __name__ == "__main__":
    env = Trading('amzn')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 64

    for e in range(EPISODES):
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
            if done:
                print("episode: {}/{}, score: {}, e: {:.5}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        # if e % 10 == 0:
# #     agent.save("./save/cartpole-dqn.h5")
