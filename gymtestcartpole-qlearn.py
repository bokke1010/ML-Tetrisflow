import gym, QLean, fileState

# Try: CartPole-v0 MountainCar-v0
# Starting with Q-learning or something
env = gym.make('CartPole-v0')
env.reset()
# print(env.observation_space.high)
# print(env.observation_space.low)
# print(env.action_space.high)
# Action space is discrete. 0 or 1


state, reward, done, debug = (0,0,0,0), 0.0, False, {}
oldState = state
model = QLean.Model(discountFactor = 0.2, learningRate = 0.06, actionList = [0, 1], randomize = 0.02)
model.data = fileState.load_state("qd.json")
counter = 0
while counter < 180:
    env.render()
    oldState = state
    action = model.bestAction(state)
    (state, reward, done, debug) = env.step(action)
    state = (int(state[0]*1.5), int(state[1]*1.2), int(state[2]*15.5), int(state[3]*0.6))
    model.updateModel(oldState, state, reward, action)
    if done:
        env.reset()
        counter = 1
    if (counter % 130) == 0:
        fileState.save_state(model.data, "qd.json")
    counter += 1
fileState.save_state(model.data, "qd.json")
env.close()
