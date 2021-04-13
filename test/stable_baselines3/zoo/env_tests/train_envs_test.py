from stable_baselines3.common.env_util import make_vec_env, make_atari_env
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack

is_eval = True
frame_stack = 4
n_envs = 8

env = make_atari_env(
    env_id='SpaceInvaders-v0',
    n_envs=n_envs,
    seed=4208325217,
    env_kwargs={},
    monitor_dir='logs',
    # wrapper_class=AtariWrapper,
    vec_env_cls=DummyVecEnv,
    vec_env_kwargs={},
    monitor_kwargs={},
)

env = VecNormalize(
    env, training=True, norm_obs=True,
    norm_reward=not is_eval, # Only normalize reward when training
    clip_obs=10.0, clip_reward=10.0,
    gamma=0.99, epsilon=1e-8
)

if frame_stack is not None:
    n_stack = frame_stack
    env = VecFrameStack(env, n_stack)
    print(f"Stacking {n_stack} frames")

# # Wrap if needed to re-order channels
# # (switch from channel last to channel first convention)
# if is_image_space(env.observation_space) and not is_image_space_channels_first(env.observation_space):
#     if self.verbose > 0:
#         print("Wrapping into a VecTransposeImage")
#     env = VecTransposeImage(env)

# # check if wrapper for dict support is needed
# if self.algo == "her":
#     if self.verbose > 0:
#         print("Wrapping into a ObsDictWrapper")
#     env = ObsDictWrapper(env)

import time
import cv2
from streamer.recorder.stream_writer import StreamWriter

stream_writer = StreamWriter(show_preview=True)

state = env.reset()
while True:
    next_state, reward, done, _ = env.step([env.action_space.sample() for i in range(n_envs)])
    print(f'next_state.shape: {next_state.shape}')
    img = env.render(mode='rgb_array')
    print(f'img.shape: {img.shape}')
    print(f'reward: {reward}')
    quit_flag = stream_writer.step(img)
    
    time.sleep(1/20)
    if quit_flag:
        cv2.imwrite('atari_wrapped.png', img)
        break