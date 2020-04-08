'''
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive trials.

Author: David Chen
'''

import gym
import math
import random
import matplotlib.pyplot as plt
from collections import defaultdict

env = gym.make('CartPole-v0')


class QlearningAlgorithm():
    def __init__(self, actions, discount, featureExtractor):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = 1
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

    def getExplorationProb(self, episode):
        if episode >= 4000:
            return 0
        return 0.4 * 0.9996 ** episode

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
    for i in range(4):
        for j in range(i + 1, 4):
            features.append(((f'f{i}{j}', state[i], state[j], action), 1))
    for i in range(4):
        features.append(((f'f{i}', state[i], action), 1))
    features.append((('same vel dir', state[1] * state[3] > 0, action), 1))
    features.append((('same pos sign', state[0] * state[2] > 0, action), 1))
    features.append((('cart moving from origin', state[0] * state[1] > 0, action), 1))
    features.append((('tip moving from origin', state[2] * state[3] > 0, action), 1))

    return features


def simulate(rl, numEpisodes=10, maxIterations=1000, verbose=False, sort=False):
    rewards = []
    t100 = 0
    for episode in range(numEpisodes):
        if episode % 100 == 0:
            print('Episode:', episode, ', Average of last 100:', t100 / 100)
            rewards.append(t100 / 100)
            if t100 / 100 >= 195:  # solved
                break
            t100 = 0
        rl.explorationProb = rl.getExplorationProb(episode)
        state = tuple(env.reset())
        totalDiscount = 1
        totalReward = 1
        for t in range(maxIterations):
            # if numEpisodes - episode < 20:
            #     env.render()
            action = rl.getAction(state)
            newState, reward, done, info = env.step(action)
            if done:
                reward = -reward

            newState = tuple(newState)

            rl.incorporateFeedback(state, action, reward, newState)
            totalReward += totalDiscount * reward
            totalDiscount *= rl.discount
            state = newState

            if done:
                t100 += t
                break

        if verbose:
            print(f'Episode {episode} timestep = {t}')
        env.close()

    return rewards


rl = QlearningAlgorithm([0, 1], .99, cartpoleFeatureExtractor)
rewards = simulate(rl, numEpisodes=5000 + 1, maxIterations=30000, verbose=False)
plt.plot(rewards)
plt.show()
