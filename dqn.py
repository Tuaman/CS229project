import gym
import numpy as np
import random

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

EPISODES = 30

class DQNAgent:
    def __init__(self, state_size, action_size, model_replay=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.9   # discount rate
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999999
        self.learning_rate = 0.0001
        self.loss = 0

        self.weights_name = "_tmp_model_weights" + str(int(np.floor(np.random.rand()*100)))
        self.model = self._build_model()
        self.model_replay = self._build_model()
        self.duplicate(self.model, self.model_replay)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        # plot_model(model, to_file='model.png')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, i = None, is_test=False):
        if not is_test and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        if i is not None and i % 20 == 0:
            print(act_values)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size, update_model_replay=True):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model_replay.predict(next_state)[0]))
            target_f = self.model.predict(state)
            predict = target_f[0][action]
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            #self.loss += (predict - target)**2
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if update_model_replay:
            self.duplicate(self.model, self.model_replay)

    def train(self, state, action, reward, next_state, done):
        #target = reward
        #if not done:
        target = (reward + self.gamma *
                  np.amax(self.model.predict(next_state)[0]))
        target_f = self.model.predict(state)
        predict = target_f[0][action]
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)
        self.loss += (predict - target)**2
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def duplicate(self, from_model, to_model):
        from_model.save_weights(self.weights_name)
        to_model.load_weights(self.weights_name)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    #agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        # if e % 10 == 0:
#     agent.save("./save/cartpole-dqn.h5")
