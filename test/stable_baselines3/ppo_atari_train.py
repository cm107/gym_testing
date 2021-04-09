import os
import time
import gym
import numpy as np
import matplotlib.pyplot as plt
import cv2
from streamer.recorder.stream_writer import StreamWriter

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class SaveCallback(BaseCallback):
    def __init__(self, save_path: str, freq: int, verbose=1):
        super(SaveCallback, self).__init__(verbose)
        self.freq = freq
        self.save_path = save_path

    def _init_callback(self) -> None:
        pass

    def _on_step(self) -> bool:
        if self.n_calls % self.freq == 0:
            self.model.save(self.save_path)

        return True

atari_env_name_list = [
    'SpaceInvaders-v0',
    'AirRaid-v0',
    'Alien-v0',
    'Amidar-v0',
    'Assault-v0',
    'Asterix-v0',
    'Asteroids-v0',
    'Atlantis-v0',
    'BankHeist-v0',
    'BattleZone-v0',
]
output_dir = 'atari_output'
os.makedirs(output_dir, exist_ok=True)
for env_name in atari_env_name_list:
    log_dir = f'{output_dir}/{env_name}'
    env = gym.make(env_name)
    callback = SaveCallback(save_path=f'{log_dir}/current.zip', freq=100)

    env0 = gym.make(env_name)
    n_frames = int(5e4)
    schedule_fn = lambda base, x: x * base
    schedule_lr = lambda x: schedule_fn(base=2.5e-4, x=x)
    schedule_clip = lambda x: schedule_fn(base=0.1, x=x)

    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=schedule_lr,
        n_steps=128,
        batch_size=32,
        n_epochs=3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=schedule_clip,
        clip_range_vf=None,
        ent_coef=0.01,
        vf_coef=1,
        tensorboard_log=log_dir,
        create_eval_env=True,
        verbose=1,
    )

    best_model_path = f'{log_dir}/best_model.zip'
    current_model_path = f'{log_dir}/current.zip'
    if not os.path.isfile(best_model_path):
        model.learn(
            total_timesteps=n_frames,
            eval_env=env0,
            eval_freq=5,
            n_eval_episodes=5,
            eval_log_path=log_dir,
            callback=callback
        )


    for path, video_filename in [(best_model_path, 'best.avi'), (current_model_path, 'current.avi')]:
        model.load(path)

        infer_duration = 60 # in seconds
        infer_fps = 30
        infer_total_frames = infer_duration * infer_fps
        stream_writer = StreamWriter(video_save_path=f'{log_dir}/{video_filename}', fps=infer_fps)
        obs = env0.reset()
        for i in range(infer_total_frames):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env0.step(action)
            img = env0.render(mode='rgb_array')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            stream_writer.step(img)
            if done:
                obs = env0.reset()
            # time.sleep(1/infer_fps)
        stream_writer.close()