import math
import numpy as np
import torch
from collections import deque
from tqdm import tqdm
from ..util.utils import get_device
from ..network.basic import BasicRLNetwork
from ..util.plot import save_score_plot

class Agent:
    def __init__(self, env, h_size=16):
        self.env = env
        self.device = get_device()
        self.network = BasicRLNetwork(
            s_size=env.observation_space.shape[0],
            h_size=h_size,
            a_size=env.action_space.shape[0]
        )
        self.network.to(self.device)
        
    def evaluate(self, weights: np.ndarray, gamma=1.0, max_t=5000, render: bool=False, show_pbar: bool=False):
        self.network.set_weights(weights)
        episode_return = 0.0
        state = self.env.reset()

        step_pbar = tqdm(total=max_t, unit='step(s)', leave=False) if show_pbar else None
        for t in range(max_t):
            state = torch.from_numpy(state).float().to(self.device)
            action = self.network.forward(state)
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

    def load(self, weights_path: str):
        self.network.load(weights_path)

    def simulate(self, weights_path: str, gamma=1.0, max_t=5000, render: bool=False, show_pbar: bool=False):
        self.load(weights_path)
        episode_return = 0.0
        state = self.env.reset()

        step_pbar = tqdm(total=max_t, unit='step(s)', leave=False) if show_pbar else None
        for t in range(max_t):
            state = torch.from_numpy(state).float().to(self.device)
            action = self.network.forward(state)
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
        self,
        n_iterations: int=500, max_t: int=1000, gamma: int=1.0, print_every: int=10, pop_size: int=50, elite_frac: float=0.2, sigma: float=0.5,
        weights_save_path: str='model.pth',
        resume: str=None,
        plot_save_path: str='score_plot.jpg'
    ):
        n_elite=int(pop_size*elite_frac)
        scores_deque = deque(maxlen=100)
        scores = []

        if resume is None:
            best_weight = sigma*np.random.randn(self.network.get_weights_dim())
        else:
            raise NotImplementedError

        iter_pbar = tqdm(total=n_iterations, unit='iter', leave=True)
        for i_iteration in tqdm(range(1, n_iterations+1), total=n_iterations, unit="iter", leave=True):
            weights_pop = [best_weight + (sigma*np.random.randn(self.network.get_weights_dim())) for i in range(pop_size)]
            
            rewards = []
            pop_pbar = tqdm(total=pop_size, unit='pop(s)', leave=False)
            pop_pbar.set_description('Accumulating Trial Rewards')
            for weights in weights_pop:
                reward = self.evaluate(weights, gamma, max_t, render=False)
                rewards.append(reward)
                pop_pbar.update()
            pop_pbar.close()
            rewards = np.array(rewards)

            elite_idxs = rewards.argsort()[-n_elite:]
            elite_weights = [weights_pop[i] for i in elite_idxs]
            best_weight = np.array(elite_weights).mean(axis=0)

            reward = self.evaluate(best_weight, gamma=1.0, max_t=max_t, render=True, show_pbar=True)
            iter_pbar.set_description(f'Latest Reward: {reward}')
            scores_deque.append(reward)
            scores.append(reward)
            
            save_score_plot(scores=scores, save_path=plot_save_path)
            torch.save(self.network.state_dict(), weights_save_path)        

            if i_iteration % print_every == 0:
                print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))

            if np.mean(scores_deque) >= 90.0:
                print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iteration-100, np.mean(scores_deque)))
                break
            iter_pbar.update()
        iter_pbar.close()
        return scores