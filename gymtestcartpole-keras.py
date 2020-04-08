import gym, fileState
import numpy as np
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents import SARSAAgent
from rl.policy import EpsGreedyQPolicy

env = gym.make('CartPole-v1')
env.reset()

# Make a neural net with 3 hidden layers
def agent(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape = (1, states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model
# Actually make a neural net with 3 hidden layers
model = agent(env.observation_space.shape[0], env.action_space.n)

policy = EpsGreedyQPolicy()
# Create a tensorflow reinforcement learning agent using the [state > action > reward] system
sarsa = SARSAAgent(model = model, policy = policy, nb_actions = env.action_space.n)
# Choose how we calculate reward and modify the model
sarsa.compile('adam', metrics = ['mse'])

# sarsa.fit(env, nb_steps = 50000, visualize = False, verbose = 1)
sarsa.load_weights('cartpolekerassarsa.h5f')

scores = sarsa.test(env, nb_episodes=10, visualize=False)
print('Average score over 10 test games: {}'.format(np.mean(scores.history['episode_reward'])))

sarsa.save_weights('cartpolekerassarsa.h5f', overwrite=True)
sarsa.test(env, nb_episodes=2, visualize=True)
