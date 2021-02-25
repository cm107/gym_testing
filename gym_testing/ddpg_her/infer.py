from .base import DDPG_HER_Base
from .lib.play import Play

class DDPG_HER_Inferer(DDPG_HER_Base):
    def __init__(
        self,
        weight_path: str,
        env_name: str="FetchPickAndPlace-v1",
    ):
        super().__init__(
            weight_path=weight_path,
            env_name=env_name,
            is_train=False
        )
    
    def run(
        self, max_episode: int=100, show_details: bool=False,
        video_save: str=None, show_preview: bool=True
    ):
        player = Play(
            env=self.env, agent=self.agent,
            max_episode=max_episode,
            weight_path=self.weight_path,
            video_save=video_save,
            show_preview=show_preview
        )
        player.evaluate(show_details=show_details)