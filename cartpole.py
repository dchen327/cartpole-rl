import gym
import math
import random
import matplotlib.pyplot as plt
from collections import defaultdict

env = gym.make('CartPole-v0')


class QlearningAlgorithm():
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions)
        else:
            # return action with max Q-value
            action = max((self.getQ(state, action), action) for action in self.actions)[1]
            return action

    def getStepSize(self):
        return 1 / math.sqrt(self.numIters)

    def incorporateFeedback(self, state, action, reward, newState):
        eta = self.getStepSize()
        V_opt = 0
        V_opt = max(self.getQ(newState, newAction) for newAction in self.actions)
        Q_opt = self.getQ(state, action)
        for f, v in self.featureExtractor(state, action):
            self.weights[f] -= eta * (Q_opt - (reward + self.discount * V_opt)) * v


def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]


def cartpoleFeatureExtractor(state, action):
    features = []
    round_pos = [2, 2, 2, 2]
    state = tuple(round(state[i], round_pos[i]) for i in range(len(state)))
    features.append((state, 1))
    features.append(((state[0], state[2]), 1))
    features.append(((state[1], state[3]), 1))
    features.append(((state[0], state[1]), 1))
    features.append(((state[2], state[3]), 1))
    features.append((('f0', state[0], action), 1))
    features.append((('f1', state[1], action), 1))
    features.append((('f2', state[2], action), 1))
    features.append((('f3', state[3], action), 1))
    features.append((('same dir', state[1] * state[3] > 0, action), 1))
    return features


def simulate(rl, numTrials=10, maxIterations=1000, verbose=False, sort=False):
    rewards = []
    r100 = 0
    for trial in range(numTrials):
        if trial % 100 == 0:
            print(trial, r100 / 100)
            rewards.append(r100 / 100)
            r100 = 0
        if trial == 2000:
            rl.explorationProb = 0.3
        if trial == 5000:
            rl.explorationProb = 0.2
        if trial == 8000:
            rl.explorationProb = 0.1
        if trial == 9500:
            rl.explorationProb = 0
        state = tuple(env.reset())
        totalDiscount = 1
        totalReward = 1
        for t in range(maxIterations):
            if numTrials - trial < 50:
                env.render()
            action = rl.getAction(state)
            newState, reward, done, info = env.step(action)
            if done:
                reward = -reward

            newState = tuple(newState)

            rl.incorporateFeedback(state, action, reward, newState)
            totalReward += totalDiscount * reward
            r100 += reward
            totalDiscount *= rl.discount
            state = newState

            if done:
                break
        if verbose:
            print(f'Trial {trial} t = {t}')
        env.close()
    return rewards


rl = QlearningAlgorithm([0, 1], .99, cartpoleFeatureExtractor, explorationProb=0.4)
rewards = simulate(rl, numTrials=10000, maxIterations=30000, verbose=False)
plt.plot(rewards)
plt.show()