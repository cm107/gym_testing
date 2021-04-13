from gym_testing.rl_adventure.worker.actor_critic import ActorCriticWorker, TrainConfig

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

atari_ram_env_name_list = [
    # 'AirRaid-ram-v0',
    # 'Alien-ram-v0',
    # 'Amidar-ram-v0',
    # 'Assault-ram-v0',
    # 'Asterix-ram-v0',
    # 'Asteroids-ram-v0',
    # 'Atlantis-ram-v0',
    # 'BankHeist-ram-v0',
    # 'BattleZone-ram-v0',
]

# uses image as observation
atari_env_name_list = [
    'SpaceInvaders-v0',
    # 'AirRaid-v0',
    # 'Alien-v0',
    # 'Amidar-v0',
    # 'Assault-v0',
    # 'Asterix-v0',
    # 'Asteroids-v0',
    # 'Atlantis-v0',
    # 'BankHeist-v0',
    # 'BattleZone-v0',
]

env_name_list = []
env_name_list += classic_control_env_name_list
env_name_list += box2d_env_name_list
env_name_list += atari_ram_env_name_list
env_name_list += atari_env_name_list

print(f'len(env_name_list): {len(env_name_list)}')

gamma = 0.99
tau = 0.95

for use_gae in [True]:
    for use_ppo in [True]:
        for env_name in env_name_list:
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
                env_name=env_name, run_id=run_id,
                output_dir='output_atari-debug-40M-atarienv',
                # output_dir='temp',
                is_atari=True
            )
            cfg = TrainConfig.get_atari_baseline_config()
            cfg.use_gae = use_gae
            cfg.use_ppo = use_ppo
            # worker.train(cfg=cfg)
            worker.infer(num_frames=1000, delay=1/20, video_save='final.avi', use_best=False, show_reward=False)
            worker.infer(num_frames=1000, delay=1/20, video_save='best.avi', use_best=True, show_reward=False)
