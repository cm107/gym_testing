import cv2
import gym
import yaml
import os
from streamer.recorder.stream_writer import StreamWriter
from stable_baselines3 import PPO

from stable_baselines3.common.utils import set_random_seed
import utils.import_envs  # noqa: F401 pylint: disable=unused-import
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from utils.exp_manager import ExperimentManager
from utils.utils import StoreDict

set_random_seed(0)

stats_path = 'logs/ppo/SpaceInvaders-v0_1/SpaceInvaders-v0'
norm_reward = False
hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=norm_reward, test_mode=True)
env_id = 'SpaceInvaders-v0'

args_path = 'logs/ppo/SpaceInvaders-v0_1/SpaceInvaders-v0/args.yml'
env_kwargs = {}
if os.path.isfile(args_path):
    with open(args_path, "r") as f:
        loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
        if loaded_args["env_kwargs"] is not None:
            env_kwargs = loaded_args["env_kwargs"]

env = create_test_env(
    env_id,
    n_envs=1,
    stats_path=stats_path,
    seed=0,
    log_dir='logs',
    should_render=True,
    hyperparams=hyperparams,
    env_kwargs=env_kwargs,
)

model = PPO.load('logs/ppo/SpaceInvaders-v0_1/best_model.zip', env=env, custom_objects={})

# infer_duration = 60 # in seconds
# infer_fps = 20
# infer_total_frames = infer_duration * infer_fps
# stream_writer = StreamWriter(video_save_path='best.avi', fps=infer_fps)
# obs = env.reset()
# for i in range(infer_total_frames):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     img = env.render(mode='rgb_array')
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     stream_writer.step(img)
#     if done:
#         obs = env.reset()
#     # time.sleep(1/infer_fps)
# stream_writer.close()