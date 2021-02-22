from __future__ import annotations
import os
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm

from gym import utils
from gym.envs.robotics import fetch_env
from gym_testing.util.utils import get_device
from gym_testing.util.plot import save_score_plot

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place.xml')


class FetchPickAndPlaceEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
        
        # Customized Variables
        self.previous_achieved_goal = None

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = fetch_env.goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        elif self.reward_type == 'dense':
            return -d
        elif self.reward_type == 'dense_on_move':
            was_moved = False
            if self.previous_achieved_goal is None:
                pass
            else:
                displacement = fetch_env.goal_distance(achieved_goal, self.previous_achieved_goal)
                if displacement > 0.0:
                    was_moved = True
            self.previous_achieved_goal = achieved_goal
            if d <= self.distance_threshold and was_moved:
                return -d
            elif d <= self.distance_threshold and not was_moved:
                return -d
            elif d > self.distance_threshold and was_moved:
                return -d * 0.5
            elif d > self.distance_threshold and not was_moved:
                return -d * 2.0
        elif self.reward_type == 'dense_on_move_else_sparse':
            was_moved = False
            if self.previous_achieved_goal is None:
                pass
            else:
                displacement = fetch_env.goal_distance(achieved_goal, self.previous_achieved_goal)
                if displacement > 0.0:
                    was_moved = True
            self.previous_achieved_goal = achieved_goal
            if was_moved:
                return -d * 0.5
            else:
                return -(d > self.distance_threshold).astype(np.float32) * 10.0
        else:
            raise ValueError(f'Invalid reward_type: {self.reward_type}')

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

class Agent(nn.Module):
    def __init__(self, env, h_size=16):
        super(Agent, self).__init__()
        self.env = env
        self.device = get_device()

        # state, hidden layer, action sizes
        self.s_size = env.observation_space['observation'].shape[0]
        self.h_size = h_size
        self.a_size = env.action_space.shape[0]
        # define layers
        self.fc1 = nn.Linear(self.s_size, self.h_size)
        self.fc2 = nn.Linear(self.h_size, self.a_size)
        
    def set_weights(self, weights):
        s_size = self.s_size
        h_size = self.h_size
        a_size = self.a_size
        # separate the weights for each layer
        fc1_end = (s_size*h_size)+h_size
        fc1_W = torch.from_numpy(weights[:s_size*h_size].reshape(s_size, h_size))
        fc1_b = torch.from_numpy(weights[s_size*h_size:fc1_end])
        fc2_W = torch.from_numpy(weights[fc1_end:fc1_end+(h_size*a_size)].reshape(h_size, a_size))
        fc2_b = torch.from_numpy(weights[fc1_end+(h_size*a_size):])
        # set the weights for each layer
        self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))
    
    def get_weights_dim(self):
        return (self.s_size+1)*self.h_size + (self.h_size+1)*self.a_size
        
    def get_weights(self) -> np.ndarray:
        pass

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x.cpu().data
        
    def evaluate(self, weights, gamma=1.0, max_t=5000, render: bool=False, show_pbar: bool=False):
        self.set_weights(weights)
        episode_return = 0.0
        state = self.env.reset()
        step_pbar = tqdm(total=max_t, unit='step(s)', leave=False) if show_pbar else None
        for t in range(max_t):
            observation = torch.from_numpy(state['observation']).float().to(self.device)
            action = self.forward(observation).numpy()
            state, reward, done, _ = self.env.step(action)
            if step_pbar is not None:
                step_pbar.set_description(f'Step Reward: {reward}')
            episode_return += reward * math.pow(gamma, t)
            if render:
                self.env.render()
            if done:
                break
            if step_pbar is not None:
                step_pbar.update()
        if step_pbar is not None:
            step_pbar.close()
        return episode_return

def train_loop(
    agent: Agent,
    n_iterations: int=500, max_t: int=1000, gamma: int=1.0, print_every: int=10, pop_size: int=50, elite_frac: float=0.2, sigma: float=0.5,
    weights_save_path: str='model.pth', plot_save_path: str='score_plot.jpg'
):
    n_elite=int(pop_size*elite_frac)

    scores_deque = deque(maxlen=100)
    scores = []

    best_weight = sigma*np.random.randn(agent.get_weights_dim())

    iter_pbar = tqdm(total=n_iterations, unit='iter', leave=True)
    for i_iteration in tqdm(range(1, n_iterations+1), total=n_iterations, unit="iter", leave=True):
        weights_pop = [best_weight + (sigma*np.random.randn(agent.get_weights_dim())) for i in range(pop_size)]
        
        rewards = []
        pop_pbar = tqdm(total=pop_size, unit='pop(s)', leave=False)
        pop_pbar.set_description('Accumulating Trial Rewards')
        for weights in weights_pop:
            reward = agent.evaluate(weights, gamma, max_t, render=False)
            rewards.append(reward)
            pop_pbar.update()
        pop_pbar.close()
        rewards = np.array(rewards)

        elite_idxs = rewards.argsort()[-n_elite:]
        elite_weights = [weights_pop[i] for i in elite_idxs]
        best_weight = np.array(elite_weights).mean(axis=0)

        reward = agent.evaluate(best_weight, gamma=1.0, max_t=max_t, render=True, show_pbar=True)
        iter_pbar.set_description(f'Latest Reward: {reward}')
        scores_deque.append(reward)
        scores.append(reward)
        
        save_score_plot(scores=scores, save_path=plot_save_path)
        torch.save(agent.state_dict(), weights_save_path)        

        if i_iteration % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))

        if np.mean(scores_deque)>=90.0:
            print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iteration-100, np.mean(scores_deque)))
            break
        iter_pbar.update()
    iter_pbar.close()
    return scores

device = get_device()
env = FetchPickAndPlaceEnv(reward_type='dense_on_move_else_sparse')
env.seed(101)
np.random.seed(101)

agent = Agent(env=env, h_size=16)
print(f"Agent Constructed")
agent = agent.to(device)
print(f"Agent Loaded To Device")

train_loop(agent=agent, max_t=50, pop_size=100)

# TODO: Need to make an Agent that has network model layers that match the observation size (25,)