import gym
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import copy
from enum import Enum

class Approach(Enum):
    random = 0,
    qlearning = 1,
    sarsa = 2

gym.envs.register(
    id='MountainCarMyEasyVersion-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=100000,      # MountainCar-v0 uses 200
    )
env = gym.make('MountainCarMyEasyVersion-v0')

#size of state and speed ranges
obsSizes = (env.observation_space.high - env.observation_space.low)

showState = False
showInterStats = False
showSplitHeatmap = False
approach = Approach.qlearning

totalState = 35
totalSpeed = 35
episodes = 50
alpha = 0.75
gamma = 0.9


binNums = [totalState, totalSpeed]
binSizes = np.divide(obsSizes, binNums)

def GenerateHeatMap():
    global totalState, totalSpeed, Q, approach
    maxQ = np.zeros((totalState, totalSpeed))
    sp, st = getBinLabels()
    ax = plt.axes()
    ax.set_title(approach.name)
    for i in range(totalSpeed):
        for j in range(totalState):
            maxQ[i, j] = np.max(Q[i, j])
    heatmap = seaborn.heatmap(maxQ, ax=ax, cmap="RdYlGn", cbar=True, xticklabels=sp, yticklabels=st)
    plt.xlabel('Speed')
    plt.ylabel('State')
    fig = heatmap.get_figure();
    fig.show()
    
def getBinLabels():
    global binSizes, binNums, env
    st = []
    sp = []
    #get the middle values of the bins
    for i in range(binNums[0]): #state
        st.append(str(round(((i*binSizes[0])-abs(env.observation_space.low[0])) + (binSizes[0]/2), 3)))
    for j in range(binNums[1]): #speed
        sp.append(str(round(((j*binSizes[1])-abs(env.observation_space.low[1])) + (binSizes[1]/2), 3)))
    return sp, st

def getBin(state):
    global binSizes, env
    shiftedState = np.add(state, abs(env.observation_space.low))
    return np.add(np.floor(np.divide(shiftedState, binSizes)), 1).astype(int) #add one for non-zero index

def getDiscreteState(state):
    global binSizes, env
    return np.subtract(np.multiply(getBin(state), binSizes), np.multiply(abs(env.observation_space.low), 2))

def getIndex(state):
    return np.subtract(getBin(state), 1) #subtract one for zero index

Q = np.zeros((binNums[0], binNums[1], env.action_space.n))
episodes += 1

print("Using {}".format(approach.name))

for episode in range(episodes):
    done = False
    R = 0
    timesteps = 0

    state = getDiscreteState(env.reset())
    while not done:
        if showState or episode == episodes-1: env.render()

        if approach == Approach.random:
            action = env.action_space.sample()
        elif approach == Approach.qlearning or Approach.sarsa:
            action = np.argmax(Q[getIndex(state)[0], getIndex(state)[1], :])

        nextState, reward, done, info = env.step(action)
        nextState = getDiscreteState(nextState)
        if approach == Approach.qlearning:
            Q[getIndex(state)[0], getIndex(state)[1], action] += alpha * \
            (reward + gamma * (np.max(Q[getIndex(nextState)[0], getIndex(nextState)[1], :]) - \
                               Q[getIndex(state)[0], getIndex(state)[1], action]))
        elif approach == Approach.sarsa:
            Q[getIndex(state)[0], getIndex(state)[1], action] += (
                    alpha * ((reward + gamma * Q[getIndex(nextState)[0], getIndex(nextState)[1], action]) - Q[getIndex(state)[0], getIndex(state)[1], action]))
            Q[getIndex(state)[0], getIndex(state)[1], action] = reward + gamma * np.max(Q[getIndex(nextState)[0], getIndex(nextState)[1]])
        R += reward
        state = nextState
        timesteps += 1
        if showState:
            print(nextState)
            print(reward)
            print(done)
        if showInterStats and episode % 100000 == 0:
            print('Episode {} Total Reward: {}'.format(episode, R))
    print("Episode {} finished after {} timesteps.".format(episode, timesteps))
env.close()
GenerateHeatMap();
