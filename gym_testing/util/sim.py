import torch
import numpy as np
from tqdm import tqdm
from collections import deque
from ..util.plot import save_score_plot
from ..agent.basic import Agent

from logger import logger
from common_utils.check_utils import check_file_exists

def cross_entropy_simulation(
    agent: Agent,
    n_iterations: int=500, max_t: int=1000, gamma: int=1.0, print_every: int=10, pop_size: int=50, elite_frac: float=0.2, sigma: float=0.5,
    weights_save_path: str='model.pth', plot_save_path: str='score_plot.jpg', resume: str=None
) -> list:
    """PyTorch implementation of the cross-entropy method.
        
    Params
    ======
        n_iterations (int): maximum number of training iterations
        max_t (int): maximum number of timesteps per episode
        gamma (float): discount rate
        print_every (int): how often to print average score (over last 100 episodes)
        pop_size (int): size of population at each iteration
        elite_frac (float): percentage of top performers to use in update
        sigma (float): standard deviation of additive noise
    """
    n_elite=int(pop_size*elite_frac)

    scores_deque = deque(maxlen=100)
    scores = []

    if resume is not None:
        check_file_exists(resume)
        state_dict = torch.load(weights_save_path)
        agent.network.load_state_dict(state_dict)
        logger.info(f"Loaded weights from {resume}")
        raise NotImplementedError
        # best_weight =
        # TODO: Finish implementing
    else:
        print(f'type(agent): {type(agent)}')
        best_weight = sigma*np.random.randn(agent.network.get_weights_dim())

    for i_iteration in tqdm(range(1, n_iterations+1), total=n_iterations, unit="iter", leave=True):
        weights_pop = [best_weight + (sigma*np.random.randn(agent.network.get_weights_dim())) for i in range(pop_size)]
        rewards = np.array([agent.evaluate(weights, gamma, max_t) for weights in weights_pop])

        elite_idxs = rewards.argsort()[-n_elite:]
        elite_weights = [weights_pop[i] for i in elite_idxs]
        best_weight = np.array(elite_weights).mean(axis=0)

        reward = agent.evaluate(best_weight, gamma=1.0)
        scores_deque.append(reward)
        scores.append(reward)
        
        save_score_plot(scores=scores, save_path=plot_save_path)
        torch.save(agent.network.state_dict(), weights_save_path)        

        if i_iteration % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))

        if np.mean(scores_deque)>=90.0:
            print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iteration-100, np.mean(scores_deque)))
            break
    return scores