from enviroment import Enviroment
from agent import DQNAgent

TOTAL_TRADING_DAYS = 300

if __name__ == "__main__":
    # initialize stock trading environment
    env = Environment(TOTAL_TRADING_DAYS ,init_money = 10000, 
        stock_name = 'amzn', first_trading_day = 30, num_history_price = 30)
    agent = DQNAgent(len(env.state))
    # Iterate the game
    for e in range(episodes):
        # reset state in the beginning of each game
        state = env.state
        # time_t represents each frame of the game
        # Our goal is to keep the pole upright as long as possible until score of 500
        # the more time_t the more score
        for time_t in range(500):
            # turn this on if you want to render
            # env.render()
            # Decide action
            action = agent.act(state)
            # Advance the game to the next frame based on the action.
            # Reward is 1 for every frame the pole survived
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)
            # make next_state the new current state for the next frame.
            state = next_state
            # done becomes True when the game ends
            # ex) The agent drops the pole
            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}"
                      .format(e, episodes, time_t))
                break
        # train the agent with the experience of the episode
        agent.replay(32)