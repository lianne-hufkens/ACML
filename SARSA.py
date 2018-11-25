import gym
import seaborn
import numpy as np

gym.envs.register(
    id='MountainCarMyEasyVersion-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=100000,  # MountainCar-v0 uses 200
)
env = gym.make('MountainCarMyEasyVersion-v0')
# size of state and speed ranges
obsSizes = (env.observation_space.high - env.observation_space.low)


totalState=35
totalSpeed=35
showState = False
showInterStats = False
episodes = 60
binNums = [totalState, totalSpeed]  # state & speed
alpha = 0.1
gamma = 0.6

binSizes = np.divide(obsSizes, binNums)



def GenerateHeatMap():
    maxQ = np.zeros((totalState, totalSpeed))
    for i in range(totalState):
        for j in range(totalSpeed):
            maxQ[i, j] = np.max(Q[i, j]);

    heatmap = seaborn.heatmap(maxQ,vmin=-10.0, vmax=0.0)
    fig = heatmap.get_figure();
    fig.savefig('D://result7')



def getBin(state):
    global binSizes
    shiftedState = np.add(state, abs(env.observation_space.low))
    return np.add(np.floor(np.divide(shiftedState, binSizes)), 1).astype(int)  # add one for non-zero index


def getDiscreteState(state):
    global binSizes
    return np.subtract(np.multiply(getBin(state), binSizes), np.multiply(abs(env.observation_space.low), 2))


def getIndex(state):
    return np.subtract(getBin(state), 1)  # subtract one for zero index


Q = np.zeros((binNums[0], binNums[1], env.action_space.n))
episodes += 1

for episode in range(episodes):
    done = False
    R = 0
    timesteps = 0

    state = getDiscreteState(env.reset())
    while not done:
        if showState or episode == episodes - 1: env.render()
        # action = env.action_space.sample() # your agent here (this takes random actions)
        action = np.argmax(Q[getIndex(state)[0], getIndex(state)[1], :])

        nextState, reward, done, info = env.step(action)
        nextState = getDiscreteState(nextState)
       # Q[getIndex(state)[0], getIndex(state)[1], action] += alpha * \
                                                           #  (reward + gamma * (np.max(
                                                            #     Q[getIndex(nextState)[0], getIndex(nextState)[1], :]) - \
                                                            #                    Q[getIndex(state)[0], getIndex(state)[
                                                            #                        1], action]))#TD(0)
        Q[getIndex(state)[0], getIndex(state)[1], action] += (
                    alpha * ((reward + gamma * Q[getIndex(nextState)[0], getIndex(nextState)[1],action]) - Q[getIndex(state)[0], getIndex(state)[1], action]));  # SARSA
        Q[getIndex(state)[0], getIndex(state)[1], action] = reward + gamma * np.max(Q[getIndex(nextState)[0], getIndex(nextState)[1]]);

        R += reward
        state = nextState
        timesteps += 1
        if showState:
            print(observation)
            print(reward)
            print(done)
        if showInterStats and episode % 100000 == 0:
            print('Episode {} Total Reward: {}'.format(episode, R))
    print("Episode {} finished after {} timesteps.".format(episode, timesteps))
env.close()
GenerateHeatMap();
# env.reset()
# print(env.action_space.n)


# env.render()
# print(env.step(env.action_space.sample()))
# print(env.observation_space)
