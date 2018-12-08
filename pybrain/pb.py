import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pybrain.rl.environments import Environment, Task
from pybrain.rl.learners.valuebased import ActionValueTable
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Q, SARSA #@UnusedImport
from pybrain.rl.experiments import Experiment

class Stock(Environment):
    # the number of action values the environment accepts
    indim = 3
    # the number of sensor values the environment produces
    outdim = 3

    # discrete state space
    discreteStates = True
    # discrete action space
    discreteActions = True

    BUY, HOLD, SELL = 0, 1, 2
    actions = [BUY, HOLD, SELL]
    # number of possible actions for discrete action space
    numActions = len(actions)

    def __init__(self, symbol, cash=10000, window=30, span=100):
        self.stock = pd.read_csv('data/'+symbol+'.us.csv')
        self.span = self.stock.shape[0] - window if span is None else span
        self.window = window
        self.cash = cash
        self.reset()

    def _observe1(self, p):
        return self.stock['Close'][self.i-self.window:self.i]

    def _observe2(self, p):
        n = np.floor(self.c/p)
        return np.array([
            [self.c-p*n,      self.n+n], # BUY
            [self.c,          self.n],   # HOLD
            [self.c+p*self.n, 0],        # SELL
        ])

    def lastPrice(self):
        return self.stock['Close'][self.i]

    def getSensors(self):
        print('Stock.getSensors', self._observe(self.lastPrice()))
        return self._observe(self.lastPrice())

    def performAction(self, action):
        print('Stock.performAction', action)
        self.c, self.n = self._observe2(self.lastPrice())[action]
        self.i += 1

    def reset(self):
        self.i = self.stock.shape[0] - self.span
        self.c = self.cash
        self.n = 0

class Trade(Task):

    def getReward(self):
        cash = self.env.c
        if self.env.i == self.env.stock.shape[0]:
            self.env.reset
        return cash

    def performAction(self, action):
        Task.performAction(self, int(action[0]))

    def getObservation(self):
        """ The agent receives its position in the maze, to make this a fully observable
            MDP problem.
        """
        obs = np.array([self.env.i-(self.env.stock.shape[0]-self.env.span)])
        return obs


if __name__ == '__main__':
    plt.gray()
    plt.ion()

    environment = Stock('amzn')

    controller = ActionValueTable(environment.span, len(Stock.actions))
    controller.initialize(1.)

    learner = Q()
    agent = LearningAgent(controller, learner)

    task = Trade(environment)

    experiment = Experiment(task, agent)

    while True:
        experiment.doInteractions(100)
        agent.learn()
        agent.reset()

        plt.pcolor(controller.params.reshape(100, 3).max(1).reshape(9,9))
        plt.show()
        plt.pause(0.1)

