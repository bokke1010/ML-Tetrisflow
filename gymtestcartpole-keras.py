import gym, fileState
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Try: CartPole-v0 MountainCar-v0
# Starting with Q-learning or something
env = gym.make('CartPole-v0')
env.reset()
# print(env.observation_space.high)
# print(env.observation_space.low)
# print(env.action_space.high)
# Action space is discrete. 0 or 1

dataset = {}
model = keras.Sequential()
# Add layers here


state, reward, done, debug = (0,0,0,0), 0.0, False, {}
# oldState = state

counter = 0
while counter < 180:
    env.render()
    # oldState = state
    # action = model.bestAction(state)

# Choose action

# Execute action, either 1 or 0
    # (state, reward, done, debug) = env.step(action)

# Add data to datasetate[2]*15.5), int(state[3]*0.6))
# Refit model

    # if done:
    #     env.reset()
    #     counter = 1
    counter += 1
fileState.save_state(dataset, "kerascartpoleData.json")
env.close()
