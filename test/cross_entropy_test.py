import gym
import numpy as np

from logger import logger

from src.util.utils import get_device
from src.network.agent import Agent
from src.util.sim import cross_entropy_simulation
from src.util.plot import save_score_plot

device = get_device()
env = gym.make('MountainCarContinuous-v0')
env.seed(101)
np.random.seed(101)

logger.info(f'env.observation_space: {env.observation_space}')
logger.info(f'env.action_space: {env.action_space}')
logger.info(f'env.action_space.low: {env.action_space.low}')
logger.info(f'env.action_space.high: {env.action_space.high}')

agent = Agent(env=env, h_size=16)
logger.info(f"Agent Constructed")
agent = agent.to(device)
logger.info(f"Agent Loaded To Device")

scores = cross_entropy_simulation(
    agent=agent,
    n_iterations=500, max_t=1000, gamma=1.0, print_every=10, pop_size=50, elite_frac=0.2, sigma=0.5,
    plot_save_path='score_plot.png', weights_save_path='model.pth', resume='model.pth'
)

# agent.load_state_dict(torch.load('checkpoint.pth'))

# state = env.reset()
# count = 0
# while True:
#     count += 1
#     print(f"count: {count}")
#     state = torch.from_numpy(state).float().to(device)
#     with torch.no_grad():
#         action = agent(state)
#         print(action)
#     env.render()
#     next_state, reward, done, _ = env.step(action)
#     state = next_state
#     if done:
#         break

# env.close()