from gym_testing.ddpg_her.infer import DDPG_HER_Inferer

inferer = DDPG_HER_Inferer(
    weight_path='/home/clayton/workspace/study/DDPG-her/Pre-trained models/FetchPickAndPlace.pth',
    env_name='FetchPickAndPlace-v1'
)
inferer.run(
    max_episode=100,
    show_details=True,
    video_save='infer_video.avi',
    show_preview=False
)