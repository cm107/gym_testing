import gym
import numpy as np
from gym_testing.agent.basic import Agent

env = gym.make('MountainCarContinuous-v0')
env.seed(101)
np.random.seed(101)

print(f'env.observation_space: {env.observation_space}')
print(f'env.action_space: {env.action_space}')
print(f'env.action_space.low: {env.action_space.low}')
print(f'env.action_space.high: {env.action_space.high}')

agent = Agent(env=env, h_size=16)

agent.train_loop(
    n_iterations=500, max_t=1000, gamma=1.0, print_every=10, pop_size=50, elite_frac=0.2, sigma=0.5,
    plot_save_path='score_plot.png',
    weights_save_path='model.pth',
    # resume='model.pth'
)