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

from enum import Enum
from typing import Tuple, Union

class Initialization(Enum):
    Zero = 0
    XavierGlorotNormal = 1
    XavierGlorotUniform = 2
    KaimingHeNormal = 3  # also known as Variance scaling
    KaimingHeUniform = 4
    Normal = 5


_init_methods = {
    Initialization.Zero: torch.zero_,
    Initialization.XavierGlorotNormal: torch.nn.init.xavier_normal_,
    Initialization.XavierGlorotUniform: torch.nn.init.xavier_uniform_,
    Initialization.KaimingHeNormal: torch.nn.init.kaiming_normal_,
    Initialization.KaimingHeUniform: torch.nn.init.kaiming_uniform_,
    Initialization.Normal: torch.nn.init.normal_,
}

def linear_layer(
    input_size: int,
    output_size: int,
    kernel_init: Initialization = Initialization.XavierGlorotUniform,
    kernel_gain: float = 1.0,
    bias_init: Initialization = Initialization.Zero,
) -> torch.nn.Module:
    """
    Creates a torch.nn.Linear module and initializes its weights.
    :param input_size: The size of the input tensor
    :param output_size: The size of the output tensor
    :param kernel_init: The Initialization to use for the weights of the layer
    :param kernel_gain: The multiplier for the weights of the kernel. Note that in
    TensorFlow, the gain is square-rooted. Therefore calling  with scale 0.01 is equivalent to calling
        KaimingHeNormal with kernel_gain of 0.1
    :param bias_init: The Initialization to use for the weights of the bias layer
    """
    layer = torch.nn.Linear(input_size, output_size)
    if (
        kernel_init == Initialization.KaimingHeNormal
        or kernel_init == Initialization.KaimingHeUniform
    ):
        _init_methods[kernel_init](layer.weight.data, nonlinearity="linear")
    else:
        _init_methods[kernel_init](layer.weight.data)
    layer.weight.data *= kernel_gain
    _init_methods[bias_init](layer.bias.data)
    return layer

def conv_output_shape(
    h_w: Tuple[int, int],
    kernel_size: Union[int, Tuple[int, int]] = 1,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> Tuple[int, int]:
    """
    Calculates the output shape (height and width) of the output of a convolution layer.
    kernel_size, stride, padding and dilation correspond to the inputs of the
    torch.nn.Conv2d layer (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
    :param h_w: The height and width of the input.
    :param kernel_size: The size of the kernel of the convolution (can be an int or a
    tuple [width, height])
    :param stride: The stride of the convolution
    :param padding: The padding of the convolution
    :param dilation: The dilation of the convolution
    """
    from math import floor

    if not isinstance(kernel_size, tuple):
        kernel_size = (int(kernel_size), int(kernel_size))
    h = floor(
        ((h_w[0] + (2 * padding) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1
    )
    w = floor(
        ((h_w[1] + (2 * padding) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1
    )
    return h, w

class SimpleVisualEncoder(nn.Module):
    def __init__(
        self, height: int, width: int, initial_channels: int, output_size: int
    ):
        super().__init__()
        self.h_size = output_size
        conv_1_hw = conv_output_shape((height, width), 8, 4)
        conv_2_hw = conv_output_shape(conv_1_hw, 4, 2)
        self.final_flat = conv_2_hw[0] * conv_2_hw[1] * 32

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=initial_channels, out_channels=16, kernel_size=[8, 8], stride=[4, 4]),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=[4, 4], stride=[2, 2]),
            nn.LeakyReLU(),
        )
        self.dense = nn.Sequential(
            linear_layer(
                input_size=self.final_flat,
                output_size=self.h_size,
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=1.41,  # Use ReLU gain
            ),
            nn.LeakyReLU(),
        )

    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        # if not exporting_to_onnx.is_exporting():
        #     visual_obs = visual_obs.permute([0, 3, 1, 2])
        hidden = self.conv_layers(visual_obs)
        hidden = hidden.reshape(-1, self.final_flat)
        return self.dense(hidden)

class VisualActorCritic(nn.Module):
    def __init__(
        self, obs_spec, num_outputs, hidden_size, std=0.0,
        is_discrete_action: bool=False
    ):
        super(VisualActorCritic, self).__init__()
        shape = obs_spec.shape
        self.vis_encoder = SimpleVisualEncoder(
            height=shape[0], width=shape[1], initial_channels=shape[2],
            output_size=hidden_size
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )

        self.is_discrete_action = is_discrete_action
        if not is_discrete_action:
            # No Softmax when working with discrete actions
            self.actor = nn.Sequential(
                nn.Linear(hidden_size, num_outputs)
            )
            self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
            self.apply(init_weights)
        else:
            self.actor = nn.Sequential(
                nn.Linear(hidden_size, num_outputs),
                nn.Softmax(dim=1),
            )

    def forward(self, x) -> (torch.Tensor, Categorical):
        hidden = self.vis_encoder(x)
        # print(f'hidden: {hidden}')
        value = self.critic(hidden) # num_envs x 1
        if not self.is_discrete_action:
            # Calculation of dist changes because there is no Softmax
            mu = self.actor(hidden)
            std = self.log_std.exp().expand_as(mu)
            dist = Normal(mu, std)
        else:
            probs = self.actor(hidden) # num_envs x num_actions
            # print(f"Flag1: probs: {probs}")
            dist  = Categorical(probs) # Distribution of action probabilities
        # print(f'dist.probs.argmax(axis=1): {dist.probs.argmax(axis=1)}')
        # print(f'dist.probs.max(axis=1): {dist.probs.max(axis=1)}')
        return dist, value # dist comes from actor (the policy), value comes from the critic (the base)

    def save(self, path: str):
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        self.load_state_dict(torch.load(path))