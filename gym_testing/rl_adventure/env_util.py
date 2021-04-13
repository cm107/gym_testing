import gym
from .multiprocessing_env import SubprocVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper

def _make_env(env_name: str, is_atari: bool=False, clip_reward: bool=True, terminal_on_life_loss: bool=True):
    def _thunk():
        env = gym.make(env_name)
        if is_atari:
            return AtariWrapper(env=env, clip_reward=clip_reward, terminal_on_life_loss=terminal_on_life_loss)
        else:
            return env

    return _thunk

def make_env(env_name: str, is_atari: bool=False, clip_reward: bool=True, terminal_on_life_loss: bool=True):
    return _make_env(env_name, is_atari=is_atari, clip_reward=clip_reward, terminal_on_life_loss=terminal_on_life_loss)()

def make_env_list(env_name: str, num_envs: int, is_atari: bool=False, clip_reward: bool=True, terminal_on_life_loss: bool=True) -> SubprocVecEnv:
    env_list = [_make_env(env_name, is_atari=is_atari, clip_reward=clip_reward, terminal_on_life_loss=terminal_on_life_loss) for i in range(num_envs)]
    env_list = SubprocVecEnv(env_list)
    return env_list

# The below methods are based on the code from baselines
from stable_baselines3.common.env_util import make_vec_env, make_atari_env
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
import random

def get_atari_env(env_id: str, training: bool=True, is_eval: bool=False, frame_stack: int=4, n_envs: int=8, log_dir: str='logs'):
    wrapper_kwargs = {
        'terminal_on_life_loss': False,
        'clip_reward': False
    } if is_eval or not training else {}
    env = make_atari_env(
        env_id=env_id,
        n_envs=n_envs,
        seed=random.randint(0, 9999999999),
        env_kwargs={},
        wrapper_kwargs=wrapper_kwargs,
        monitor_dir=log_dir,
        vec_env_cls=DummyVecEnv,
        vec_env_kwargs={},
        monitor_kwargs={},
    )
    env = VecNormalize(
        env, training=training, norm_obs=True,
        norm_reward=not is_eval, # Only normalize reward when training
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99, epsilon=1e-8
    )
    if frame_stack is not None:
        n_stack = frame_stack
        env = VecFrameStack(env, n_stack)
        print(f"Stacking {n_stack} frames")
    return env