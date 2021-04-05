from gym_testing.rl_adventure.worker.actor_critic import ActorCriticWorker

classic_control_env_name_list = [
    # 'Acrobot-v1',
    # 'CartPole-v1',
    # 'MountainCar-v0',
    # 'MountainCarContinuous-v0',
    # 'Pendulum-v0',
]

box2d_env_name_list = [
    # 'BipedalWalker-v3',
    # 'BipedalWalkerHardcore-v3',
    # # 'CarRacing-v0', # uses image as observation
    # 'LunarLander-v2',
    # 'LunarLanderContinuous-v2'
]

atari_env_name_list = [
    'AirRaid-ram-v0',
    # 'AirRaid-v0', # uses image as observation
    'Alien-ram-v0',
    # 'Alien-v0', # uses image as observation
    'Amidar-ram-v0',
    # 'Amidar-v0', # uses image as observation
    'Assault-ram-v0',
    # 'Assault-v0', # uses image as observation
    'Asterix-ram-v0',
    # 'Asterix-v0' # uses image as observation
    'Asteroids-ram-v0',
    # 'Asteroids-v0', # uses image as observation
    'Atlantis-ram-v0',
    # 'Atlantis-v0', # uses image as observation
    'BankHeist-ram-v0',
    # 'BankHeist-v0', # uses image as observation
    'BattleZone-ram-v0',
    # 'BattleZone-v0', # uses image as observation
]

env_name_list = []
env_name_list += classic_control_env_name_list
env_name_list += box2d_env_name_list
env_name_list += atari_env_name_list

print(f'len(env_name_list): {len(env_name_list)}')

gamma = 0.99
tau = 0.95

for env_name in env_name_list:
    for use_gae in [False, True]:
        for use_ppo in [False, True]:
            run_id = f"gamma{gamma}-tau{tau}"
            if use_gae:
                run_id = f"{run_id}-gae"
            else:
                run_id = f"{run_id}-no_gae"
            if use_ppo:
                run_id = f"{run_id}-ppo"
            else:
                run_id = f"{run_id}-no_ppo"

            worker = ActorCriticWorker(
                env_name=env_name, run_id=run_id, output_dir='output-ppo_comparison'
            )
            worker.train(
                num_steps=100, max_frames=20000*20,
                gamma=gamma, tau=tau,
                use_gae=use_gae, use_ppo=use_ppo,
                max_train_time=15 # in minutes
            )
            # worker.infer(num_frames=500, delay=1/20, video_save='infer.avi')
