import cv2
import numpy as np
import time
from gym_testing.rl_adventure.env_util import make_env
from streamer.recorder.stream_writer import StreamWriter

stream_writer = StreamWriter(show_preview=True)
env = make_env(env_name='SpaceInvaders-v0', is_atari=True, clip_reward=False, terminal_on_life_loss=False)

state = env.reset()

while True:
    next_state, reward, done, _ = env.step(env.action_space.sample())
    img = env.render(mode='rgb_array')
    img = env.observation(img)
    print(f'img.shape: {img.shape}')
    quit_flag = stream_writer.step(img)
    
    time.sleep(1/20)
    if quit_flag:
        cv2.imwrite('atari_wrapped.png', img)
        break