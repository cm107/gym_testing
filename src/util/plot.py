from logger import logger
import matplotlib.pyplot as plt
from common_utils.path_utils import get_dirpath_from_filepath, rel_to_abs_path
from common_utils.check_utils import check_dir_exists

def save_score_plot(
    scores: list, save_path: str, preview: bool=False
):
    episodes = [i+1 for i in range(len(scores))]
    
    plt.title(label='score vs episodes')
    plt.plot(
        episodes, scores,
        label='scores', color='blue'
    )
    plt.legend(loc='lower right')
    plt.xlabel('episodes')
    plt.ylabel('score')
    min_episode = min(episodes)
    max_episode = max(episodes)
    min_score = min(scores)
    max_score = max(scores)
    plt.xlim(min_episode - 1, max_episode + 1)

    score_interval = max_score - min_score
    bound_offset = score_interval * 0.05
    plt.ylim(min_score - bound_offset, max_score * bound_offset)

    if save_path is not None:
        abs_save_path = rel_to_abs_path(save_path)
        save_dir = get_dirpath_from_filepath(filepath=abs_save_path)
        check_dir_exists(save_dir)
        plt.savefig(abs_save_path)
        plt.clf()
    else:
        logger.error(f"Please specify a save_path.")
        raise Exception
    if preview:
        plt.show()