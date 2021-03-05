from gym_testing.ddpg_her.train import DDPG_HER_Trainer
from gym_testing.ddpg_her.infer import DDPG_HER_Inferer
from common_utils.file_utils import make_dir_if_not_exists, delete_all_files_in_dir

env_name_list = [
    # 'FetchPickAndPlace-v1',
    # 'FetchPush-v1',
    # 'FetchReach-v1',
    # 'FetchSlide-v1',
    'HandManipulateBlock-v0',
    'HandManipulateEgg-v0',
    'HandManipulatePen-v0',
    'HandReach-v0'
]
output_root_dir = 'robotics_output'
make_dir_if_not_exists(output_root_dir)

for env_name in env_name_list:
    for max_epochs in [2000]:
        for lr, lr_name in [(1e-3, '1e-3'), (5e-4, '5e-4'), (1e-4, '1e-4')]:
            target_dirname = f'epoch{max_epochs}_lr{lr_name}'
            target_output_dir = f'{output_root_dir}/{target_dirname}'
            make_dir_if_not_exists(target_output_dir)

            env_root_dir = f'{target_output_dir}/{env_name}'
            make_dir_if_not_exists(env_root_dir)
            delete_all_files_in_dir(env_root_dir, ask_permission=False)
            log_dir = f'{env_root_dir}/logs'
            plot_path = f'{env_root_dir}/plot.png'
            weight_path = f'{env_root_dir}/model.pth'
            infer_video_path = f'{env_root_dir}/infer_video.avi'

            print(f'========{env_name}=========')
            trainer = DDPG_HER_Trainer(
                weight_path=weight_path,
                env_name=env_name,
                actor_lr=lr, critic_lr=lr
            )
            trainer.train(
                max_epochs=max_epochs, max_cycles=50,
                max_episodes=2, num_updates=40,
                n_timesteps=50, nontrivial_goal_thresh=0.05,
                penalize_actions=True,
                plot_save_path=plot_path,
                plot_include=['success', 'actor', 'critic'],
                log_dir=log_dir
            )
            inferer = DDPG_HER_Inferer(
                weight_path=weight_path,
                env_name=env_name
            )
            inferer.run(
                max_episode=100,
                show_details=True,
                video_save=infer_video_path,
                show_preview=False
            )