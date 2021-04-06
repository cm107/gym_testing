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
    'AirRaid-v0',
    'Alien-v0',
    'Amidar-v0',
    'Assault-v0',
    'Asterix-v0'
    'Asteroids-v0',
    'Atlantis-v0',
    'BankHeist-v0',
    'BattleZone-v0',
]

# TODO: Try Space Invaders
# Using Visual Encoder at https://arxiv.org/pdf/1312.5602v1.pdf
"""
We now describe the exact architecture used for all seven Atari games. The input to the neural
network consists is an 84 × 84 × 4 image produced by φ. The first hidden layer convolves 16 8 × 8
filters with stride 4 with the input image and applies a rectifier nonlinearity [10, 18]. The second
hidden layer convolves 32 4 × 4 filters with stride 2, again followed by a rectifier nonlinearity. The
final hidden layer is fully-connected and consists of 256 rectifier units. The output layer is a fullyconnected linear layer with a single output for each valid action.
The number of valid actions varied
between 4 and 18 on the games we considered. We refer to convolutional networks trained with our
approach as Deep Q-Networks (DQN).
"""

env_name_list = []
env_name_list += classic_control_env_name_list
env_name_list += box2d_env_name_list
env_name_list += atari_ram_env_name_list
env_name_list += atari_env_name_list

print(f'len(env_name_list): {len(env_name_list)}')

gamma = 0.99
tau = 0.95

for env_name in env_name_list:
    for use_gae in [True, False]:
        for use_ppo in [True, False]:
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
                output_dir='output_atari-debug'
            )
            worker.train(
                num_steps=100, max_frames=20000*20,
                gamma=gamma, tau=tau,
                use_gae=use_gae, use_ppo=use_ppo,
                num_envs=16,
                max_train_time=15 # in minutes
            )
            worker.infer(num_frames=500, delay=1/20, video_save='infer.avi')
