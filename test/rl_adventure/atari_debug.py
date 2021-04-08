from gym_testing.rl_adventure.worker.actor_critic import ActorCriticWorker, TrainConfig

worker = ActorCriticWorker(
    env_name='SpaceInvaders-v0', run_id='0', output_dir='output-debug'
)
cfg = TrainConfig.get_atari_config()
# worker.train(cfg=cfg)
worker.infer(num_frames=500, delay=1/20, video_save='final.avi', use_best=False)
worker.infer(num_frames=500, delay=1/20, video_save='best.avi', use_best=True)

# import gym
# import time
# env = gym.make('SpaceInvaders-v0')
# state = env.reset()

# count = 0
# while True:
#     # action = env.action_space.sample()
#     action = (int(count / 10)) % 6
#     count += 1
#     next_state, reward, done, _ = env.step(action)
#     print(action)
#     print(f'reward: {reward}')
#     env.render()
#     time.sleep(1/10)
#     if done:
#         break
# env.close()
