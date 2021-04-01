import torch
from torch import nn
from torch.distributions import Categorical, Normal

def init_weights(m, mean: float=0, std: float=0.1, bias: float=0.1):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=mean, std=std)
        nn.init.constant_(m.bias, val=bias)

class ActorCritic(nn.Module):
    def __init__(
        self, num_inputs, num_outputs, hidden_size, std=0.0,
        is_discrete_action: bool=False
    ):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.is_discrete_action = is_discrete_action
        if not is_discrete_action:
            # No Softmax when working with discrete actions
            self.actor = nn.Sequential(
                nn.Linear(num_inputs, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_outputs)
            )
            self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
            self.apply(init_weights)
        else:
            self.actor = nn.Sequential(
                nn.Linear(num_inputs, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_outputs),
                nn.Softmax(dim=1),
            )

    def forward(self, x) -> (torch.Tensor, Categorical):
        # print(f"Flag0: x: {x}")
        value = self.critic(x) # num_envs x 1
        if not self.is_discrete_action:
            # Calculation of dist changes because there is no Softmax
            mu = self.actor(x)
            std = self.log_std.exp().expand_as(mu)
            dist = Normal(mu, std)
        else:
            probs = self.actor(x) # num_envs x num_actions
            # print(f"Flag1: probs: {probs}")
            dist  = Categorical(probs) # Distribution of action probabilities
        return dist, value # dist comes from actor (the policy), value comes from the critic (the base)

    def save(self, path: str):
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        self.load_state_dict(torch.load(path))