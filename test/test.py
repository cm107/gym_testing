import gym
import numpy as np
from logger import logger
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
        logger.blue(ordered_list)

tracker = ValueTracker()

env = gym.make("CartPole-v1")
logger.purple(f"env: {env}")
logger.purple(f"env.action_space: {env.action_space}")
logger.purple(f"env.observation_space: {env.observation_space}")
logger.purple(f"env.observation_space.low: {env.observation_space.low}")
logger.purple(f"env.observation_space.high: {env.observation_space.high}")
logger.blue(f"env.action_space.contains(0): {env.action_space.contains(0)}")
logger.cyan(f"gym.envs.registry.all():\n{gym.envs.registry.all()}")
import sys
sys.exit()

observation = env.reset()
for _ in range(100):
    env.render()
    action = env.action_space.sample() # your agent here (this takes random actions)
    logger.purple(action)
    tracker.add(action)
    observation, reward, done, info = env.step(action)
    logger.cyan(f"observation: {observation}")
    logger.cyan(f"reward: {reward}")
    logger.cyan(f"done: {done}")
    logger.cyan(f"info: {info}")

    if done:
        observation = env.reset()
env.close()

tracker.summary()