import QLean, gym, fileState

env = gym.make('Pendulum-v0')
env.reset()

# print(env.action_space.high)
# print(env.action_space.low)
# print(env.observation_space.high)
# print(env.observation_space.high)
# data = fileState.load_state("1pend.txt"),
model = QLean.Model(data = fileState.load_state("1pend.txt"), learningRate = 0.1, discountFactor = 0.2, actionList = [env.action_space.low, env.action_space.low*0.5, env.action_space.low*0, env.action_space.high*0.5, env.action_space.high])
state, oldState = (0,0,0), None
rewardStack = -16

while rewardStack < -2:
    oldState = state
    env.render()
    action = model.bestAction(state)
    (state, reward, done, debug) = env.step(action)
    state = (int(state[0]*4), int(state[1]*4), int(state[2]))
    model.updateModel(oldState, state, reward, action)
    if done:
        env.reset()
    rewardStack *= 0.97
    rewardStack += 0.03 * reward
    print(rewardStack)
fileState.save_state(model.data, "1pend.txt")
env.close()
