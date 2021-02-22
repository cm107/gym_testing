from __future__ import annotations
from typing import List
import random
import os
import numpy as np
from gym import utils
from gym.envs.robotics import fetch_env
from common_utils.common_types.point import Point3D

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

env = FetchPickAndPlaceEnv()
print(f'env.action_space: {env.action_space}')
print(f'env.action_space.low: {env.action_space.low}')
print(f'env.action_space.high: {env.action_space.high}')
print(f'env.observation_space: {env.observation_space}')


# state = env.reset()

class RobotArmAction:
    def __init__(self, x: float, y: float, z: float, g: float):
        for val in [x, y, z, g]:
            assert val >= -1.0 and val <= 1.0
        self.x = x
        self.y = y
        self.z = z
        self.g = g
    
    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z, self.g]
    
    @classmethod
    def random(cls, include: List[str]=['x', 'y', 'z', 'g']) -> RobotArmAction:
        return RobotArmAction(
            x=random.uniform(-1.0, 1.0) if 'x' in include else 0.0,
            y=random.uniform(-1.0, 1.0) if 'y' in include else 0.0,
            z=random.uniform(-1.0, 1.0) if 'z' in include else 0.0,
            g=random.uniform(-1.0, 1.0) if 'g' in include else 0.0
        )

class RobotArmState:
    def __init__(self, achieved_goal: Point3D, desired_goal: Point3D, observation: np.ndarray):
        self.achieved_goal = achieved_goal
        self.desired_goal = desired_goal
        self.observation = observation
    
    def to_dict(self) -> dict:
        # Note: Not JSON-able
        return {
            'achieved_goal': self.achieved_goal.to_numpy(),
            'desired_goal': self.desired_goal.to_numpy(),
            'observation': self.observation
        }
    
    @classmethod
    def from_dict(cls, item_dict: dict) -> RobotArmState:
        return RobotArmState(
            achieved_goal=Point3D.from_numpy(item_dict['achieved_goal']),
            desired_goal=Point3D.from_numpy(item_dict['desired_goal']),
            observation=item_dict['observation']
        )

# Action Space: [dx, dy, dz, dg]
# Where (dx, dy, dz) refers to the displacement of the robot arm and dg refers to the displacement of the gripper fingers.
while True:
    # state = torch.from_numpy(state).float().to(device)
    # with torch.no_grad():
    #     action = agent(state)
    #     print(action)
    env.render()
    # next_state, reward, done, _ = env.step(action)
    next_state, reward, done, _ = env.step(RobotArmAction.random(include=['x', 'y']).to_list())
    next_state = RobotArmState.from_dict(next_state)
    # print(next_state)
    # state = next_state
    if done:
        break

env.close()