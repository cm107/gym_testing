import gym
import numpy as np

from gym_testing.util.utils import get_device
from gym_testing.agent.basic import Agent

device = get_device()
env = gym.make('MountainCarContinuous-v0')
env.seed(101)
np.random.seed(101)

agent = Agent(env=env, h_size=16)
agent.network = agent.network.to(device)

agent.simulate(
    weights_path='model.pth', gamma=1.0,
    max_t=5000, render=True, show_pbar=True
)