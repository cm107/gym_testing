import gym
import numpy as np
from common_utils.counter_utils import count_list_items

class ValueTracker:
    def __init__(self):
        self.value_list = []

    def add(self, value):
        self.value_list.append(value)

    def mean(self, axis: int=0):
        return np.array(self.value_list).mean(axis=axis)

    def summary(self):
        summary_str = ""
        ordered_list = count_list_items(self.value_list)
        print(ordered_list)

tracker = ValueTracker()

env = gym.make("CartPole-v1")
print(f"env: {env}")
print(f"env.action_space: {env.action_space}")
print(f"env.observation_space: {env.observation_space}")
print(f"env.observation_space.low: {env.observation_space.low}")
print(f"env.observation_space.high: {env.observation_space.high}")
print(f"env.action_space.contains(0): {env.action_space.contains(0)}")
print(f"gym.envs.registry.all():\n{gym.envs.registry.all()}")
import sys
sys.exit()

observation = env.reset()
for _ in range(100):
    env.render()
    action = env.action_space.sample() # your agent here (this takes random actions)
    print(action)
    tracker.add(action)
    observation, reward, done, info = env.step(action)
    print(f"observation: {observation}")
    print(f"reward: {reward}")
    print(f"done: {done}")
    print(f"info: {info}")

    if done:
        observation = env.reset()
env.close()

tracker.summary()