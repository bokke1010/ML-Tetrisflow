import gym

from gym import envs
print(envs.registry.all())

# Try: CartPole-v0 MountainCar-v0
# Additional dependencies for MsPacman-v0 Hopper-v1
# Starting with Q-learning or something
env = gym.make('CartPole-v0')
env.reset()
print(env.observation_space.high)
print(env.observation_space.low)
# Action space is discrete 0 or 1
state, reward, done, debug = [0,0,0,0], 0.0, False, {}
action = 0
for _ in range(200):
    env.render()
    # action = env.action_space.sample()
    action = 1-action
    (state, reward, done, debug) = env.step(action)
    if done:
        env.reset()
    # print(state)
    # Returns object with state data, reward, done (won or failed), debug object
env.close()
