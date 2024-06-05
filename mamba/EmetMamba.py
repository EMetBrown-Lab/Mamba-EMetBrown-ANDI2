import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba import MambaConfig
from mamba_ssm import Mamba


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

    conv_stack: int = 2  # How many convolutial stack at the beggining
    dropout: float = 0.05  # how much dropout in the Add&Norm layer of the bi-Mamba

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)


# Define a dataclass for the EmetMamba model
@dataclass
class EmetMamba(nn.Module):
    # Define the model parameters

    # Initialize the model
    def __post_init__(self):
        super().__init__()
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments

        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)

        self.config = MambaConfig(
            self.d_model,
            self.n_layers,
            self.dt_rank,
            self.d_state,
            self.expand_factor,
            self.d_conv,
            self.dt_min,
            self.dt_max,
            self.dt_init,
            self.dt_scale,
            self.rms_norm_eps,
            self.bias,
            self.conv_bias,
            self.inner_layernorms,
            self.pscan,
            self.use_cuda,
        )

    # Define the convolutional stack
    def _convolutional_stack(self):

        return nn.Sequential(
            *[
                convolutional_bloc(self.d_model, self.d_conv, self.conv_bias)
                for _ in range(self.conv_stack)
            ]
        )

    # Define the forward pass
    def forward(self, x):

        convolutional_stack = self._convolutional_stack()

        x = convolutional_stack(x)

        x2 = x.clone().detach()

        mamba = Mamba2(MambaConfig)

        x = mamba(x)

        x2 = x2

        return x


# Define a dataclass for the convolutional block
@dataclass
class convolutional_bloc(nn.Module):
    # Define the block parameters
    d_inner: int
    d_conv: int
    conv_bias: bool

    # Initialize the block
    def __post_init__(self):
        super().__init__()

        # Initialization of the 1D convolution layer
        self.conv1d()

    # Define the 1D convolution layer
    def conv1d(self):
        self.conv = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            bias=self.conv_bias,
            groups=self.d_inner,
            padding=self.d_conv - 1,
        )

    # Define the forward pass
    def forward(self, x):
        # Get the size of the input tensor
        _, L, _ = x.size()  # B,L,D

        # Initialize the instance normalization layer
        self.IN_norm = nn.InstanceNorm1d(L)

        # Save the initial input tensor
        x_start = x.clone().detach()  # B,L,D

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
        x = x + x_start  # B,D,L

        # Return the output tensor
        return x


class BiMamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

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

        self.forward_norm = nn.LayerNorm(self.config.d_model, bias=self.config.bias).to(
            self.config.device
        )

        self.backward_norm = nn.LayerNorm(
            self.config.d_model, bias=self.config.bias
        ).to(self.config.device)

        self.final_norm = nn.LayerNorm(self.config.d_model, bias=self.config.bias).to(
            self.config.device
        )
        self.dropout = nn.Dropout(self.config.dropout).to(self.config.device)

        self.feed_forward = nn.Sequential(
            nn.Linear(
                self.config.d_model, self.config.d_model * self.config.expand_factor
            ),
            nn.GELU(),
            nn.Linear(
                self.config.d_model * self.config.expand_factor, self.config.d_model
            ),
        ).to(self.config.device)

        ## False bias -> No learnable weights in this layer
        ## it reduces the risks of over fitting

    def forward(self, x):
        # x = x.to("cuda")
        residual = x

        ## Forward Branch

        y_forward_mamba = self.forward_mamba(x)

        ## Add & norm of the forward mamba
        y_forward_mamba = self.dropout(y_forward_mamba + residual)
        y_forward_mamba = self.forward_norm(y_forward_mamba)


        ##Backward Branch

        y__backward_mamba = torch.flip(x, [1])  # Flipping only the time dim
        y__backward_mamba = self.backward_mamba(y__backward_mamba)
        y__backward_mamba = torch.flip(y__backward_mamba, [1])

        ## Add & norm of the backward mamba
        y__backward_mamba = y__backward_mamba + residual
        y__backward_mamba = self.dropout(y__backward_mamba)
        y__backward_mamba = self.forward_norm(y__backward_mamba)

        ## Reconnect branches

        y = y_forward_mamba + y__backward_mamba

        residual = y  # change the residual the last add & norm after the feed forward

        ## Making it pass thourg a feedforward

        y = self.feed_forward(y)

        ## Last Add & norm

        y = self.dropout(y + residual)

        y = self.final_norm(y)

        return y


if __name__ == "__main__":

    B, L, D = 40, 100, 3
    x = torch.randn(B, L, D).to("cuda")
    config = EmetConfig(D,L)
    model = BiMamba(config)

    y = model(x)

    assert(y.size() == x.size())