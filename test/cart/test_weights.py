import gym
import numpy as np
from gym_testing.agent.basic import Agent

env = gym.make('MountainCarContinuous-v0')
env.seed(101)
np.random.seed(101)
agent = Agent(env=env, h_size=16)

print(f'agent.network.state_dict():\n{agent.network.state_dict()}')

weights = agent.network.get_weights0()
print(f'Before:\n{weights}')
weights['fc1.weight'] = np.zeros_like(weights['fc1.weight'])
print(f'weights:\n{weights}')
agent.network.set_weights0(weights, device=agent.device)
weights = agent.network.get_weights0()
print(f'After:\n{weights}')