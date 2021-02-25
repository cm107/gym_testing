from gym_testing.ddpg_her.infer import DDPG_HER_Inferer

inferer = DDPG_HER_Inferer(
    weight_path='HandManipulateBlock-v0.pth',
    env_name='HandManipulateBlock-v0'
)
inferer.run(
    max_episode=100,
    show_details=True,
    video_save='infer_video.avi',
    show_preview=True
)