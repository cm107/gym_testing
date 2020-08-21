import torch

from logger import logger

def get_device() -> torch.device:
    use_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(use_device)
    logger.purple(f"type(device): {type(device)}")
    return device

def load_to_device(device, obj, dtype):
    obj.to(device=device, dtype=dtype)