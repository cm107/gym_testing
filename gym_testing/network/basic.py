import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from collections import OrderedDict

class BasicRLNetwork(nn.Module):
    def __init__(self, s_size: int, h_size: int, a_size: int):
        """Basic Reinforcement Learning Network

        Args:
            s_size (int): State size
            h_size (int): Hidden size
            a_size (int): Action size
        """
        super().__init__()
        
        self.s_size = s_size
        self.h_size = h_size
        self.a_size = a_size

        # define layers
        self.fc1 = nn.Linear(self.s_size, self.h_size)
        self.fc2 = nn.Linear(self.h_size, self.a_size)
    
    def load(self, weights_path: str):
        state_dict = torch.load(weights_path)
        self.load_state_dict(state_dict=state_dict, strict=True)

    def set_weights(self, weights: np.ndarray):
        # separate the weights for each layer
        fc1_end = (self.s_size*self.h_size)+self.h_size
        fc1_W = torch.from_numpy(weights[:self.s_size*self.h_size].reshape(self.s_size, self.h_size))
        fc1_b = torch.from_numpy(weights[self.s_size*self.h_size:fc1_end])
        fc2_W = torch.from_numpy(weights[fc1_end:fc1_end+(self.h_size*self.a_size)].reshape(self.h_size, self.a_size))
        fc2_b = torch.from_numpy(weights[fc1_end+(self.h_size*self.a_size):])
        
        # set the weights for each layer
        self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))

    # def get_weights0(self) -> Dict[str, np.ndarray]:
    #     result = OrderedDict()
    #     for k, v in self.state_dict().items():
    #         result[k] = v.cpu().numpy()
    #     return result

    # def set_weights0(self, weights: Dict[str, np.ndarray], device):        
    #     for k, v in weights.items():
    #         val = torch.from_numpy(v).to(device)
    #         layer_name = k.replace('.weight', '').replace('.bias', '')
    #         if k.endswith('.weight'):
    #             attr_name = 'weight'
    #         elif k.endswith('.bias'):
    #             attr_name = 'bias'
    #         else:
    #             raise Exception
    #         self.__dict__['_modules'][layer_name].__dict__['_parameters'][attr_name] = val

    # def init_weights(self):
    #     def func(m):
    #         if type(m) == nn.Linear:
    #             nn.init.xavier_uniform(m.weight, gain=1.0)
    #             m.bias.data.fill_(0.01)

    #     self.apply(func)

    def get_weights_dim(self) -> int:
        return (self.s_size+1)*self.h_size + (self.h_size+1)*self.a_size
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x.cpu().data