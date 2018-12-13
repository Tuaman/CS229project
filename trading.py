import numpy as np
import pandas as pd

from grapher import Grapher
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
        self.symbol = symbol
        self.stock = pd.read_csv('data/'+symbol+'.us.csv')
        self.init = {'span': span, 'cash': cash, 'start':start}
        self.window = window

        self.action_labels = ['BUY', 'HOLD', 'SELL']
        self.action_space = spaces.Discrete(len(self.action_labels))
        self.done_action = self.action_labels.index('SELL')

        low = np.zeros(self.window+3)
        high = np.array([np.finfo(np.float32).max]*(self.window+3)) # TODO limit num remaining trading days
        self.observation_space = spaces.Box(low, high, dtype=np.float32)


        self.state = None

    def get_state(self, i=None, cash=None, nown=None):
        i    = i    or self.i
        cash = cash or self.holdings[0]
        nown = nown or self.holdings[1]

        state = np.hstack([[i, cash, nown], self.history()])
        assert self.observation_space.shape[0] == state.shape[0], "%d (expected %d) invalid state size"%(state.shape[0], self.observation_space.shape[0])

        return state

    def today(self):
        return self.start+self.init['span']-self.i

    def price(self):
        return self.stock['Close'][self.today()]

    def history(self):
        return self.stock['Close'][self.today()-self.window:self.today()].values

    def observations(self, cash, nown, p):
        n = np.floor(cash/p)
        return np.array([
            [cash-p*n,      nown+n], # BUY
            [cash,          nown],   # HOLD
            [cash+p*nown, 0],        # SELL
        ])

    def eval(self, c, n):
        return c + n*self.price()

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        asset_before = self.eval(*self.holdings)
        done = (self.i == 0)
        if not done:
            self.i -= 1
        else:
            action = self.done_action

        self.holdings = self.observations(self.holdings[0], self.holdings[1], self.price())[action]
        self.state = self.get_state()

        if not done:
            reward = self.eval(*self.holdings) - asset_before
        else:
            reward = 0

        return np.array(self.state), reward, done, {}

    def reset(self):
        # Internal states
        self.i = self.init['span']
        self.holdings = np.array([self.init['cash'], 0])

        self.start = self.init['start'] or np.random.randint(self.window, self.stock.shape[0]-3*self.init['span']-1)
        self.state = self.get_state()
        return self.state

if __name__ == "__main__":
    env = Trading('amzn')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 32

    title = env.symbol.upper()+' '+os.path.basename(__file__).split('.')[0]
    grapher = Grapher(env.symbol.upper(), env.action_labels)

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
            grapher.add(cash, nown, price, action, reward)

            state = next_state
            if done:
                print('start', env.start, 'previous', (cash, nown), 'current', tuple(env.holdings))
                print("episode: {}/{}, score: {}, e: {:.5}"
                      .format(e, EPISODES, time, agent.epsilon))
                grapher.show(ep=e, t=time, e=agent.epsilon)
                grapher.reset()
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        # if e % 10 == 0:
# #     agent.save("./save/cartpole-dqn.h5")
