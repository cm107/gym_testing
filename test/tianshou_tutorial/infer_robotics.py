import os
assert os.environ.get("LD_PRELOAD") is not None, "Please set LD_PRELOAD."
import gym
import torch
import pickle
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import DQNPolicy
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, ReplayBuffer, PrioritizedReplayBuffer

from gym_testing.env.robotics import FetchPickAndPlaceEnv

task = 'pick_and_place'
seed = 1000
hidden_sizes = [128, 128, 128, 128]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 1e-3
gamma = 0.9
n_step = 3
target_update_freq = 320
eps_test = 0.05
render = 1 / 100 # 1 / FPS

# env = gym.make(task)
# state_shape = env.observation_space.shape or env.observation_space.n
# action_shape = env.action_space.shape or env.action_space.n
reward_type = 'dense_on_move_else_sparse'
env = FetchPickAndPlaceEnv(reward_type=reward_type)
state_shape = env.observation_space['observation'].shape or env.observation_space['observation'].n
action_shape = env.action_space.shape or env.action_space.n

# seed
np.random.seed(seed)
torch.manual_seed(seed)

# model
net = Net(
    state_shape, action_shape,
    hidden_sizes=hidden_sizes, device=device,
).to(device)
optim = torch.optim.Adam(net.parameters(), lr=lr)
policy = DQNPolicy(
    net, optim, gamma, n_step,
    target_update_freq=target_update_freq
)
policy.load_state_dict(torch.load(f'log/{task}/dqn0/policy.pth'))

# Let's watch its performance!
policy.eval()
policy.set_eps(eps_test)
collector = Collector(policy, env)
result = collector.collect(n_episode=1, render=render)
print(f'Final reward: {result["rew"]}, length: {result["len"]}')