import os
from typing import List
import gym
import torch
import pickle
import pprint
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import DQNPolicy
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, ReplayBuffer, PrioritizedReplayBuffer

from ...env.robotics import FetchPickAndPlaceEnv

class FetchPickAndPlaceTrainer:
    def __init__(
        self,
        seed: int=1626,
        lr: float=1e-3, gamma: float=0.9,
        n_step: int=3, target_update_freq: int=320,
        hidden_sizes: List[int]=[128, 128, 128, 128],
        training_num: int=8, test_num: int=100,
        logdir: str='log',
        device: str=None
    ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # self.reward_type = 'dense_on_move_else_sparse'
        self.reward_type = 'custom'
        env = self._get_env()
        self.reward_threshold = env.spec.reward_threshold
        state_shape = env.observation_space['observation'].shape or env.observation_space['observation'].n
        # TODO: env.observation_space isn't the same as env.observation_space["observation"], and this is breaking the API.
        # Figure out how to fix this.
        action_shape = env.action_space.shape or env.action_space.n
        # you can also use tianshou.env.SubprocVectorEnv
        self.training_num, self.test_num = training_num, test_num
        self.train_envs = DummyVectorEnv(
            [lambda: self._get_env() for _ in range(training_num)])
        self.test_envs = DummyVectorEnv(
            [lambda: self._get_env() for _ in range(test_num)])
        
        # seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.train_envs.seed(seed)
        self.test_envs.seed(seed)

        # Q_param = V_param = {"hidden_sizes": [128]}
        # model
        net = Net(state_shape, action_shape,
                    hidden_sizes=hidden_sizes, device=device,
                    # dueling=(Q_param, V_param),
                    ).to(device)
        optim = torch.optim.Adam(net.parameters(), lr=lr)
        self.policy = DQNPolicy(
            net, optim, gamma, n_step,
            target_update_freq=target_update_freq)
        
        # log
        self.log_path = os.path.join(logdir, 'pick_and_place', 'dqn')
        self.writer = SummaryWriter(self.log_path)

    def _get_env(self) -> FetchPickAndPlaceEnv:
        return FetchPickAndPlaceEnv(reward_type=self.reward_type)

    def train(
        self,
        epoch: int=10, step_per_epoch: int=1000,
        eps_test: float=0.05, eps_train: float=0.1,
        collect_per_step: int=10, batch_size: int=64,
        prioritized_replay: bool=False, buffer_size: int=20000,
        alpha: float=0.6, beta: float=0.4,
        render: float=0.,
    ):
        # buffer
        if prioritized_replay:
            buf = PrioritizedReplayBuffer(
                buffer_size, alpha=alpha, beta=beta)
        else:
            buf = ReplayBuffer(buffer_size)

        # collector
        train_collector = Collector(self.policy, self.train_envs, buf)
        test_collector = Collector(self.policy, self.test_envs)

        # policy.set_eps(1)
        train_collector.collect(n_step=batch_size)

        def save_fn(policy):
            torch.save(self.policy.state_dict(), os.path.join(self.log_path, 'policy.pth'))

        def stop_fn(mean_rewards):
            print(f'mean_rewards: {mean_rewards}')

            return mean_rewards >= self.reward_threshold

        def train_fn(epoch, env_step):
            # eps annnealing, just a demo
            if env_step <= 10000:
                self.policy.set_eps(eps_train)
            elif env_step <= 50000:
                eps = eps_train - (env_step - 10000) / \
                    40000 * (0.9 * eps_train)
                self.policy.set_eps(eps)
            else:
                self.policy.set_eps(0.1 * eps_train)

        def test_fn(epoch, env_step):
            self.policy.set_eps(eps_test)

        # trainer
        result = offpolicy_trainer(
            self.policy, train_collector, test_collector, epoch,
            step_per_epoch, collect_per_step, self.test_num,
            batch_size, train_fn=train_fn, test_fn=test_fn,
            stop_fn=stop_fn, save_fn=save_fn, writer=self.writer)

        # assert stop_fn(result['best_reward'])
        if stop_fn(result['best_reward']):
            print('Success!')
        else:
            print(f"Failure: result['best_reward'] == {result['best_reward']} < {env.spec.reward_threshold}")

        pprint.pprint(result)
        # Let's watch its performance!
        env = self._get_env()
        self.policy.eval()
        self.policy.set_eps(eps_test)
        collector = Collector(self.policy, env)
        result = collector.collect(n_episode=1, render=render)
        print(f'Final reward: {result["rew"]}, length: {result["len"]}')

        # save buffer in pickle format, for imitation learning unittest
        buf = ReplayBuffer(buffer_size)
        collector = Collector(self.policy, self.test_envs, buf)
        collector.collect(n_step=buffer_size)
        pickle.dump(buf, open('FetchPickAndPlaceEnv.pkl', "wb"))