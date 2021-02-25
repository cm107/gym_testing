import random
import numpy as np
import torch
import gym
from mpi4py import MPI
from copy import deepcopy as dc
from common_utils.file_utils import file_exists

from .lib.agent import Agent

class DDPG_HER_Base:
    def __init__(
        self,
        weight_path: str,
        env_name: str="FetchPickAndPlace-v1",
        memory_size: float=7e+5//50,
        batch_size: int=256,
        actor_lr: float=1e-3,
        critic_lr: float=1e-3,
        gamma: float=0.98,
        tau: float=0.05,
        k_future: int=4,
        is_train: bool=False
    ):
        if not file_exists(weight_path) and not is_train:
            raise FileNotFoundError(f"Couldn't find weights at: {weight_path}")
        self.weight_path = weight_path
        self.env = gym.make(env_name)
        self.env.seed(MPI.COMM_WORLD.Get_rank())
        random.seed(MPI.COMM_WORLD.Get_rank())
        np.random.seed(MPI.COMM_WORLD.Get_rank())
        torch.manual_seed(MPI.COMM_WORLD.Get_rank())
        self.agent = Agent(
            n_states=self.env.observation_space.spaces["observation"].shape,
            n_actions=self.env.action_space.shape[0],
            n_goals=self.env.observation_space.spaces["desired_goal"].shape[0],
            action_bounds=[self.env.action_space.low[0], self.env.action_space.high[0]],
            capacity=memory_size,
            action_size=self.env.action_space.shape[0],
            batch_size=batch_size,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            tau=tau,
            k_future=k_future,
            env=dc(self.env)
        )