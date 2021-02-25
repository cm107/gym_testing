import torch
from torch import device
import numpy as np
import cv2
from gym import wrappers
from mujoco_py import GlfwContext
from streamer.recorder.stream_writer import StreamWriter
from common_utils.cv_drawing_utils import draw_text_rows_in_corner
import time

GlfwContext(offscreen=True)

from mujoco_py.generated import const


class Play:
    def __init__(self, env, agent, max_episode=4, weight_path: str="FetchPickAndPlace.pth", video_save: str=None, show_preview: bool=True, use_monitor: bool=False):
        self.env = env
        if use_monitor:
            self.env = wrappers.Monitor(env, "./videos", video_callable=lambda episode_id: True, force=True)
        self.max_episode = max_episode
        self.agent = agent
        self.agent.load_weights(weight_path)
        self.agent.set_to_eval_mode()
        self.device = device("cuda" if torch.cuda.is_available() else "cpu")
        if video_save is not None or show_preview:
            self.stream_writer = StreamWriter(video_save_path=video_save, fps=30, show_preview=show_preview)
        else:
            self.stream_writer = None

    def evaluate(self, show_details: bool=False):
        for _ in range(self.max_episode):
            env_dict = self.env.reset()
            state = env_dict["observation"]
            achieved_goal = env_dict["achieved_goal"]
            desired_goal = env_dict["desired_goal"]
            while np.linalg.norm(achieved_goal - desired_goal) <= 0.05:
                env_dict = self.env.reset()
                state = env_dict["observation"]
                achieved_goal = env_dict["achieved_goal"]
                desired_goal = env_dict["desired_goal"]
            done = False
            episode_reward = 0
            quit_flag = False
            while not done:
                action = self.agent.choose_action(state, desired_goal, train_mode=False)
                next_env_dict, r, done, _ = self.env.step(action)
                next_state = next_env_dict["observation"]
                next_desired_goal = next_env_dict["desired_goal"]
                episode_reward += r
                state = next_state.copy()
                desired_goal = next_desired_goal.copy()
                I = self.env.render(mode="rgb_array")
                self.env.viewer.cam.type = const.CAMERA_FREE
                self.env.viewer.cam.fixedcamid = 0
                I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
                if show_details:
                    row_text_list = [
                        f'episode_reward: {episode_reward}',
                        f'achieved_goal: {[round(val, 2) for val in next_env_dict["achieved_goal"]]}',
                        f'desired_goal: {[round(val, 2) for val in next_env_dict["desired_goal"]]}',
                        f'action: {[round(val, 2) for val in action]}'
                    ]
                    I = draw_text_rows_in_corner(
                        img=I,
                        row_text_list=row_text_list,
                        row_height=I.shape[0]*0.04,
                        color=[255, 0, 0]
                    )
                if self.stream_writer is not None:
                    quit_flag0 = self.stream_writer.step(I)
                    if not quit_flag and quit_flag0:
                        quit_flag = True
                    if self.stream_writer.viewer is not None:
                        time.sleep(0.002)
            if quit_flag:
                break
        if self.stream_writer is not None:
            self.stream_writer.close()

        self.env.close()
