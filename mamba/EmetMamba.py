import math
from dataclasses import dataclass
from typing import Union
from copy import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EmetConfig:
    d_model: int  # D
    n_layers: int
    dt_rank: Union[int, str] = "auto"
    d_state: int = 16  # N in paper/comments
    expand_factor: int = 2  # E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    rms_norm_eps: float = 1e-5

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False  # apply layernorms to internal activations

    pscan: bool = True  # use parallel scan mode or sequential mode when training
    use_cuda: bool = (
        False  # use official CUDA implementation when training (not compatible with (b)float16)
    )

    bi_mamba_stacks: int = 1  # How many bi mamba to stack
    conv_stack: int = 2  # How many convolutial stack at the beggining
    dropout: float = 0.05  # how much dropout in the Add&Norm layer of the bi-Mamba

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("importing gpu Mamba")
            from mamba_ssm import Mamba
        else:            
            print("importing cpu Mamba")
            from .mamba import Mamba

        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)


# Define a dataclass for the EmetMamba model
@dataclass(eq=False)
class EmetMamba(nn.Module):
    # Define the model parameters

    config: EmetConfig

    # Initialize the model
    def __post_init__(self):
        super().__init__()
        self.d_inner = (
            self.config.expand_factor * self.config.d_model
        )  # E*D = ED in comments

        if self.config.dt_rank == "auto":
            self.config.dt_rank = math.ceil(self.d_model / 16)

        self._convolutional_stack()
        self._bi_mamba_stacks()

        self.out_proj_s = nn.Sequential(
            Feed_foward(
                self.config.d_model,
                self.config.d_inner,
                self.config,
            ),
            nn.Linear(self.config.d_model, 3, bias=False),
        ).to(self.config.device)

        self.out_proj_D_a = nn.Sequential(
            Feed_foward(
                self.config.d_model + 1,
                (self.config.d_model + 1) * self.config.expand_factor,
                self.config,
            ),
            nn.Linear(self.config.d_model + 1, 2, bias=False),
        ).to(self.config.device)

    # Define the convolutional stack
    def _convolutional_stack(self):

        self.convolutional_stack_input = nn.Sequential(
            *[convolutional_bloc(self.config) for _ in range(self.config.conv_stack)]
        ).to(self.config.device)

        self.convolutional_stack_alpha_D = nn.Sequential(
            *[convolutional_bloc(self.config) for _ in range(self.config.conv_stack)]
        ).to(self.config.device)

    # Define the biMamba stacks
    def _bi_mamba_stacks(self):

        self.bi_mamba_stacks_s = nn.Sequential(
            *[BiMamba(self.config) for _ in range(self.config.bi_mamba_stacks)]
        )

        self.bi_mamba_plus = nn.Sequential(
            *[Bi_mamba_plus(self.config) for _ in range(self.config.bi_mamba_stacks)]
        )

    # Define the forward pass
    def forward(self, x):

        x = torch.flatten(x, start_dim=1, end_dim=2)
        copy_x = x

        # Making input pass through the convolutional stack

        x_conved = self.convolutional_stack_input(x)
        x = self.bi_mamba_stacks_s(x_conved) #

        s_probas = self.out_proj_s(x)  #  Here we should out put a (B,L,num_classes) output

        return s_probas
        # s = torch.argmax(torch.softmax(s_probas, dim = 2), dim=2) # getting the actual top probable classes
        # s = s.unsqueeze(-1) # adding a dimension for the next step

        # concat_entry = torch.cat((copy_x, s), dim=2)

        # out_bimamba_plus = self.bi_mamba_plus(concat_entry)

        # alpha_d_a = self.out_proj_D_a(out_bimamba_plus)

        # # Concat final ouput

        # # output = torch.cat((alpha_d_a, s), dim=2)

        # return [s_probas, alpha_d_a]


# Define a dataclass for the convolutional block
@dataclass(eq=False)
class convolutional_bloc(nn.Module):
    # Define the block parameters
    config: EmetConfig

    # Initialize the block
    def __post_init__(self):
        super().__init__()

        # Initialization of the 1D convolution layer
        self.conv1d()

    # Define the 1D convolution layer
    def conv1d(self):
        self.conv = nn.Conv1d(
            in_channels=self.config.d_model,
            out_channels=self.config.d_model,
            kernel_size=self.config.d_conv,
            bias=self.config.conv_bias,
            groups=self.config.d_model,
            padding=self.config.d_conv - 1,
        ).to(self.config.device)

    # Define the forward pass
    def forward(self, x):
        # Get the size of the input tensor
        _, L, _ = x.size()  # B,L,D

        # Initialize the instance normalization layer
        self.IN_norm = nn.InstanceNorm1d(L).to(self.config.device)

        # Save the initial input tensor
        residual = x  # B,L,D

        # Transpose the input tensor
        x = x.transpose(1, 2)  # B,D,L

        # Apply the 1D convolution layer
        x = self.conv(x)[:, :, :L]  # B,D,L using a short filter

        # Transpose the output tensor
        x = x.transpose(1, 2)  # B,D,L

        # Apply the instance normalization layer
        x = self.IN_norm(x)  # B,D,L

        # Apply the leaky ReLU activation function
        x = F.leaky_relu(x)  # B,D,L

        # Add the initial input tensor
        x = x + residual  # B,D,L

        # Return the output tensor
        return x


class BiMamba(nn.Module):
    def __init__(self, config):
        # Initialize the BiMamba module with a given configuration
        super().__init__()
        self.config = config

        if torch.cuda.is_available():
            from mamba_ssm import Mamba
        else:
            from Mamba import Mamba

        # Initialize two Mamba modules for forward and backward passes
        self.forward_mamba = Mamba(
            d_model=self.config.d_model,
            d_state=self.config.n_layers,
            d_conv=self.config.d_conv,
            expand=self.config.expand_factor,
        ).to(self.config.device)

        self.backward_mamba = Mamba(
            d_model=self.config.d_model,
            d_state=self.config.n_layers,
            d_conv=self.config.d_conv,
            expand=self.config.expand_factor,
        ).to(self.config.device)

        # Initialize layer normalization modules for forward, backward, and final passes
        self.forward_norm = nn.LayerNorm(self.config.d_model, bias=self.config.bias).to(
            self.config.device
        )

        self.backward_norm = nn.LayerNorm(
            self.config.d_model, bias=self.config.bias
        ).to(self.config.device)

        self.final_norm = nn.LayerNorm(self.config.d_model, bias=self.config.bias).to(
            self.config.device
        )

        # Initialize a dropout module
        self.dropout = nn.Dropout(self.config.dropout).to(self.config.device)



        # Note: Setting bias=False in the layer normalization modules reduces the risk of overfitting

    def forward(self, x):

        _,L,_ = x.size()
        # Initialize a feed-forward module with two linear layers and a GELU activation function

        self.feed_forward = nn.Sequential(
            nn.Linear(
                self.config.d_model, self.config.d_model * self.config.expand_factor
            ),
            nn.ReLU(),
            nn.Linear(
                self.config.d_model * self.config.expand_factor, self.config.d_model
            ),
        ).to(self.config.device)

        # Define the forward pass through the BiMamba module
        residual = x

        # Forward branch
        y_forward_mamba = self.forward_mamba(x)
        y_forward_mamba = self.dropout(y_forward_mamba + residual)
        y_forward_mamba = self.forward_norm(y_forward_mamba)

        # Backward branch
        y_backward_mamba = torch.flip(x, [1])  # Flip the time dimension
        y_backward_mamba = self.backward_mamba(y_backward_mamba)
        y_backward_mamba = torch.flip(y_backward_mamba, [1])
        y_backward_mamba = y_backward_mamba + residual
        y_backward_mamba = self.dropout(y_backward_mamba)
        y_backward_mamba = self.forward_norm(y_backward_mamba)

        # Reconnect branches
        y = y_forward_mamba + y_backward_mamba

        residual = y  # Update the residual for the feed-forward module

        # Feed-forward module
        y = self.feed_forward(y)

        # Final add and norm
        y = self.dropout(y + residual)
        y = self.final_norm(y)

        return y


class Feed_foward(nn.Module):
    def __init__(self, in_dim, hidden_dim, config):
        super().__init__()
        self.config = config
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.in_dim),
        ).to(device=self.config.device)

    def forward(self, x):
        return self.net(x)


class Bi_mamba_plus(nn.Module):

    def __init__(self, config: EmetConfig):
        super().__init__()
        self.config = copy(config)

        self.config.d_model = self.config.d_model + 1  # adding s
        self.config.d_inner = self.config.d_model * self.config.expand_factor

        self.feed_forward = Feed_foward(
            self.config.d_model, self.config.d_inner, self.config
        )

        self.bi_mamba = BiMamba(self.config).to(device=self.config.device)

        self.dropout = nn.Dropout(p=self.config.dropout).to(self.config.device)

        self.normalization = nn.InstanceNorm1d(self.config.d_model, device=self.config.device)

    def forward(self, x):

        residual = x

        x = self.bi_mamba(x)

        x = self.dropout(x + residual)

        x = self.feed_forward(x)

        return x


# if __name__ == "__main__":

    # B, L, D = 40, 100, 3
    # config = EmetConfig(D, L)
    # x = torch.randn(B, L, D).to(config.device)
    # model = EmetMamba(config)

    # y = model(x)
    # print(x.size())
    # print(y[0].size())
    # print(y[1].size())

    # assert y.size() == x.size()
