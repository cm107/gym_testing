import gym
import cv2
import numpy as np
from streamer.cv_viewer import SimpleVideoViewer
from common_utils.cv_drawing_utils import draw_text_rows_in_corner
import math

def calc_goal_distance(env_dict: dict) -> float:
    achieved_goal = env_dict["achieved_goal"]
    desired_goal = env_dict["desired_goal"]
    return np.linalg.norm(achieved_goal - desired_goal)

def reset_until_goal(env, thresh: float=0.05) -> dict:
    env_dict = env.reset()
    while calc_goal_distance(env_dict) <= thresh:
        env_dict = env.reset()
    return env_dict

viewer = SimpleVideoViewer(preview_width=1000)

env = gym.make('FetchPickAndPlace-v1')
n_actions = env.action_space.shape[0]
env_dict = env.reset()
next_env_dict = None

count = 0
iter_per_idx = 100

while True:
    idx = int(count/iter_per_idx) % n_actions
    iter_idx = count % iter_per_idx
    count += 1
    # action = np.random.rand(n_actions)
    action = np.zeros(n_actions)
    # action[idx:idx+1] = np.random.rand(1)
    action[idx:idx+1] = np.array([math.sin((iter_idx/iter_per_idx)*2*math.pi)])
    if next_env_dict is not None:
        env_dict = next_env_dict.copy()
    next_env_dict, r, done, _ = env.step(action)
    
    state = env_dict["observation"]
    achieved_goal = env_dict["achieved_goal"]
    desired_goal = env_dict["desired_goal"]
    print(f'state:\n{state}')
    print(f'achieved_goal:\n{achieved_goal}')
    print(f'desired_goal:\n{desired_goal}')
    goal_distance = calc_goal_distance(env_dict)
    print(f'goal_distance:\n{goal_distance}')

    I = env.render(mode="rgb_array")
    I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
    text_rows = [
        f'idx: {idx}'
    ]
    I = draw_text_rows_in_corner(
        img=I,
        row_text_list=text_rows,
        row_height=I.shape[0]*0.05
    )
    quit_flag = viewer.show(I)
    if quit_flag:
        break
env.close()
viewer.close()