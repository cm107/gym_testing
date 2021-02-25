from gym_testing.ddpg_her.train import DDPG_HER_Trainer
from gym_testing.ddpg_her.infer import DDPG_HER_Inferer
from common_utils.file_utils import make_dir_if_not_exists, delete_all_files_in_dir

env_name_list = [
    'FetchPickAndPlace-v1',
    'FetchPush-v1',
    'FetchReach-v1',
    'FetchSlide-v1',
    'HandManipulateBlock-v0',
    'HandManipulateEgg-v0',
    'HandManipulatePen-v0',
    'HandReach-v0'
]
output_root_dir = 'robotics_output'
make_dir_if_not_exists(output_root_dir)

for env_name in env_name_list:
    env_root_dir = f'{output_root_dir}/{env_name}'
    make_dir_if_not_exists(env_root_dir)
    delete_all_files_in_dir(env_root_dir, ask_permission=False)
    log_dir = f'{env_root_dir}/logs'
    plot_path = f'{env_root_dir}/plot.png'
    weight_path = f'{env_root_dir}/model.pth'
    infer_video_path = f'{env_root_dir}/infer_video.avi'

    print(f'========{env_name}=========')
    trainer = DDPG_HER_Trainer(
        weight_path=weight_path,
        env_name=env_name
    )
    trainer.train(
        max_epochs=200, max_cycles=50,
        max_episodes=2, num_updates=40,
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