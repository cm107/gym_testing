import cv2
import gym
import numpy as np
from streamer.cv_viewer import cv_simple_image_viewer

def calc_goal_distance(env_dict: dict) -> float:
    achieved_goal = env_dict["achieved_goal"]
    desired_goal = env_dict["desired_goal"]
    return np.linalg.norm(achieved_goal - desired_goal)

def reset_until_nontrivial_goal(env, thresh: float=0.05) -> dict:
    env_dict = env.reset()
    while calc_goal_distance(env_dict) <= thresh:
        env_dict = env.reset()
    return env_dict

def view(env):
    I = env.render(mode="rgb_array")
    I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
    cv_simple_image_viewer(I)

# env = gym.make('HandManipulateBlock-v0')
env = gym.make("FetchPickAndPlace-v1")
env_dict = reset_until_nontrivial_goal(env=env, thresh=0.05) # Reset until greater than 5cm away.
state = env_dict["observation"]
achieved_goal = env_dict["achieved_goal"]
desired_goal = env_dict["desired_goal"]
print(f'state:\n{state}')
print(f'achieved_goal:\n{achieved_goal}')
print(f'desired_goal:\n{desired_goal}')
goal_distance = calc_goal_distance(env_dict)
print(f'goal_distance:\n{goal_distance}')
view(env)