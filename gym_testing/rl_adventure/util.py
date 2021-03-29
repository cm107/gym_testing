import os
import torch

def get_device() -> torch.device:
    use_cuda = torch.cuda.is_available()
    return torch.device("cuda" if use_cuda else "cpu")

def make_dir_if_not_exists(path: str):
    if not os.path.isdir(path):
        os.mkdir(path)