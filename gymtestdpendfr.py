import gym, math
# from tensorflow import keras

# from gym import envs
# print(envs.registry.all())

env = gym.make('InvertedDoublePendulum-v2')
env.reset()
print(env.observation_space)
print(env.action_space)
# Action space is discrete 0 or 1
state, reward, done, debug = None, 0.0, False, {}
action = [0]
magnitude = 1
t = 0
for _ in range(800):
    env.render()
    t += 0.05
    action[0] = magnitude * math.sin(math.pi*t)
    # action = env.action_space.sample()
    (state, reward, done, debug) = env.step(action)
    if done:
        env.reset()
    print(reward)
    # Returns object with state data, reward, done (won or failed), debug object
env.close()
