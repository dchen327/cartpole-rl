# Cartpole
Q-learning solution to [OpenAI's Cartpole](https://gym.openai.com/envs/CartPole-v0/)

## Introduction
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive trials.

## Q-learning with Function Approximation
Q-learning is a reinforcement learning algorithm which seeks to discover the best action to take given a state. It sometimes takes random actions to explore, and new data is used to improve future performance. Linear function approximation allows Q-learning to generalize to unseen states; however, the features used are handpicked and have a large impact on performance.

![cartpole](https://user-images.githubusercontent.com/37674516/78819170-cb8f0780-79a3-11ea-8ad6-069968da4d14.gif)

This program utilizes many concepts from Stanford CS221, and is able to solve the problem in around 5000 episodes. Better hyperparameter tuning is required to improve performance, and using a neural network instead of linear function approximation may also be helpful.