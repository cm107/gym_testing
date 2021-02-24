import os
import numpy as np
from gym import utils
from gym.envs.robotics import fetch_env

class EnvSpec:
    def __init__(self, reward_threshold: float=100):
        self.reward_threshold = reward_threshold

class DisplacementBuffer:
    def __init__(self, buffer_size: int=100, target_displacement: float=0.25, max_reward: float=1.0):
        self.buffer = []
        self.max_displacement = None
        self.min_displacement = None
        self.target_displacement = target_displacement
        self.max_reward = max_reward
    
    def process(self, displacement: float):
        if len(self.buffer) < 20:
            self.buffer.append(displacement)
        else:
            del self.buffer[0]
            self.buffer.append(displacement)
        if self.max_displacement is None or displacement > self.max_displacement:
            self.max_displacement = displacement
        if self.min_displacement is None or displacement < self.min_displacement:
            self.min_displacement = displacement
    
    @property
    def buffer_min(self) -> float:
        return min(self.buffer) if len(self.buffer) > 0 else None
    
    @property
    def buffer_max(self) -> float:
        return max(self.buffer) if len(self.buffer) > 0 else None

    def print_summary(self):
        print(f'Record Min,Max: ({self.min_displacement},{self.max_displacement}), Recent Min,Max: ({self.buffer_min},{self.buffer_max})')
    
    @property
    def working_target_displacement(self) -> float:
        return self.target_displacement if self.max_displacement is None or self.max_displacement < self.target_displacement else self.max_displacement

    def calc_reward(self) -> float:
        if self.buffer_max is not None:
            return self.max_reward * (self.buffer_max / self.max_displacement)
        else:
            return 0.0

class FetchPickAndPlaceEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', t_max: int=1000):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        # Ensure we get the path separator correct on windows
        MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place.xml')
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
        
        # Customized Variables
        self.previous_achieved_goal = None
        self.t_max = t_max
        self.t_current = 0
        self.spec = EnvSpec(
            reward_threshold=100
        )

        # Debug
        self.displacement_buff = DisplacementBuffer(buffer_size=100) # displacement usually between 0 and 0.3

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
                self.displacement_buff.process(displacement)
                # self.displacement_buff.print_summary()
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
        elif self.reward_type == 'custom':
            if self.previous_achieved_goal is None:
                pass
            else:
                displacement = fetch_env.goal_distance(achieved_goal, self.previous_achieved_goal)
                self.displacement_buff.process(displacement)
                if displacement > 0.0:
                    was_moved = True
            self.previous_achieved_goal = achieved_goal
            reward = -d + self.displacement_buff.calc_reward()
            # print(f'{-d} + {self.displacement_buff.calc_reward()}: {reward}')
            
            return reward
        else:
            raise ValueError(f'Invalid reward_type: {self.reward_type}')

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        self.t_current += 1
        # print(f'{self.t_current}/{self.t_max}')
        done = False if self.t_current < self.t_max else True
        if done:
            self.t_current = 0
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        # return obs, reward, done, info
        return obs['observation'], reward, done, info