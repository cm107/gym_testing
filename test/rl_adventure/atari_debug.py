# from gym_testing.rl_adventure.worker.actor_critic import ActorCriticWorker

# worker = ActorCriticWorker(
#     env_name='AirRaid-ram-v0', run_id='0', output_dir='output-atari_debug'
# )
# worker.train(
#     num_steps=100, max_frames=20000*20,
#     gamma=0.99, tau=0.95,
#     use_gae=True, use_ppo=True,
#     max_train_time=15 # in minutes
# )
# worker.infer(num_frames=500, delay=1/20, video_save='infer.avi')

import gym
import time
env = gym.make('SpaceInvaders-v0')
state = env.reset()

count = 0
while True:
    # action = env.action_space.sample()
    action = (int(count / 10)) % 6
    count += 1
    next_state, reward, done, _ = env.step(action)
    print(action)
    print(f'reward: {reward}')
    env.render()
    time.sleep(1/10)
    if done:
        break
env.close()