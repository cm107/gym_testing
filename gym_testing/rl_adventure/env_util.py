import gym
from .multiprocessing_env import SubprocVecEnv

def _make_env(env_name: str):
    def _thunk():
        env = gym.make(env_name)
        return env

    return _thunk

def make_env(env_name: str):
    return _make_env(env_name)()

def make_env_list(env_name: str, num_envs: int) -> SubprocVecEnv:
    env_list = [_make_env(env_name) for i in range(num_envs)]
    env_list = SubprocVecEnv(env_list)
    return env_list