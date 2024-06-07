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
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)


    # Initialize the model
    def __post_init__(self):
        # Call the parent's __init__ method
        super().__init__()
        
        # Calculate the inner dimension (d_inner) based on the expand factor and model dimension
        self.d_inner = (
            self.config.expand_factor * self.config.d_model
        )  # E*D = ED in comments

        # If dt_rank is set to "auto", calculate it based on the model dimension
        if self.config.dt_rank == "auto":
            self.config.dt_rank = math.ceil(self.d_model / 16)

        # Initialize the convolutional stack and biMamba stacks
        self._convolutional_stack()
        self._bi_mamba_stacks()

        # Define the output projection layers for s and alpha_D_a
        self.out_proj_s = nn.Sequential(
            # Feed-forward layer with d_model input, d_inner hidden, and d_model output
            Feed_foward(
                self.config.d_model,
                self.config.d_inner,
                self.config,
            ),
            # Linear layer with d_model input, 1 output, and no bias
            nn.Linear(self.config.d_model, 1, bias=False),
        ).to(self.config.device)

        self.out_proj_D_a = nn.Sequential(
            # Feed-forward layer with d_model + 1 input, (d_model + 1) * expand_factor hidden, and d_model + 1 output
            Feed_foward(
                self.config.d_model + 1,
                (self.config.d_model + 1) * self.config.expand_factor,
                self.config,
            ),
            # Linear layer with d_model + 1 input, 2 output, and no bias
            nn.Linear(self.config.d_model + 1, 2, bias=False),
        ).to(self.config.device)

    # Define the convolutional stack
    def _convolutional_stack(self):
        # Initialize the convolutional stack for input and alpha_D
        self.convolutional_stack_input = nn.Sequential(
            *[convolutional_bloc(self.config) for _ in range(self.config.conv_stack)]
        )

        self.convolutional_stack_alpha_D = nn.Sequential(
            *[convolutional_bloc(self.config) for _ in range(self.config.conv_stack)]
        )

    # Define the biMamba stacks
    def _bi_mamba_stacks(self):
        # Initialize the biMamba stacks for s and biMamba_plus
        self.bi_mamba_stacks_s = nn.Sequential(
            *[BiMamba(self.config) for _ in range(self.config.bi_mamba_stacks)]
        )

        self.bi_mamba_plus = nn.Sequential(
            *[Bi_mamba_plus(self.config) for _ in range(self.config.bi_mamba_stacks)]
        )

    # Define the forward pass
    def forward(self, x):
        # Make a copy of the input
        copy_x = x

        # Pass the input through the convolutional stack
        x_conved = self.convolutional_stack_input(x)

        # Pass the convolved input through the biMamba stacks for s
        x = self.bi_mamba_stacks_s(x_conved)

        # Output s using the output projection layer
        s = self.out_proj_s(x)  

        # Concatenate the input and s along dimension 2
        concat_entry = torch.cat((copy_x, s), dim=2)

        # Pass the concatenated input through the biMamba_plus stack
        out_bimamba_plus = self.bi_mamba_plus(concat_entry)

        # Output alpha_D_a using the output projection layer
        alpha_d_a = self.out_proj_D_a(out_bimamba_plus)

        # Concatenate alpha_D_a and s along dimension 2 as the final output
        output = torch.cat((alpha_d_a, s), dim=2)

        return output


# Define a dataclass for the convolutional block
@dataclass
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
        self.IN_norm = nn.InstanceNorm1d(L, device=self.config.device)

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

        # Dynamically import Mamba module based on CUDA availability
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

        # Note: Setting bias=False in the layer normalization modules reduces the risk of overfitting

    def forward(self, x):
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
        # Initialize the feed-forward module with input dimension, hidden dimension, and configuration
        super().__init__()
        self.config = config
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # Define the feed-forward network as a sequence of linear and ReLU layers
        self.net = nn.Sequential(
            # Linear layer with input dimension, hidden dimension, and default initialization
            nn.Linear(self.in_dim, self.hidden_dim),
            # ReLU activation function
            nn.ReLU(),
            # Linear layer with hidden dimension, input dimension, and default initialization
            nn.Linear(self.hidden_dim, self.in_dim),
        ).to(device=self.config.device)

    def forward(self, x):
        # Define the forward pass through the feed-forward module
        return self.net(x)

class Bi_mamba_plus(nn.Module):
    def __init__(self, config: EmetConfig):
        # Initialize the Bi_mamba_plus module with a given configuration
        super().__init__()
        self.config = copy(config)

        # Update the model dimensions by adding 1 to d_model and calculating d_inner
        self.config.d_model = self.config.d_model + 1  # adding s
        self.config.d_inner = self.config.d_model * self.config.expand_factor

        # Initialize the feed-forward module with d_model, d_inner, and config
        self.feed_forward = Feed_foward(
            self.config.d_model, self.config.d_inner, self.config
        )

        # Initialize the BiMamba module with the updated config and move it to the specified device
        self.bi_mamba = BiMamba(self.config).to(device=self.config.device)

        # Initialize a dropout module with the specified dropout probability and move it to the specified device
        self.dropout = nn.Dropout(p=self.config.dropout).to(self.config.device)

        # Initialize an instance normalization module with L dimensions and move it to the specified device
        self.normalization = nn.InstanceNorm1d(L, device=self.config.device)

    def forward(self, x):
        # Define the forward pass through the Bi_mamba_plus module
        residual = x

        # Pass the input through the BiMamba module
        x = self.bi_mamba(x)

        # Apply dropout to the output and add the residual
        x = self.dropout(x + residual)

        # Pass the output through the feed-forward module
        x = self.feed_forward(x)

        return x


if __name__ == "__main__":

    B, L, D = 40, 100, 3
    config = EmetConfig(D, L)
    x = torch.randn(B, L, D).to(config.device)
    model = EmetMamba(config)

    y = model(x)
    print(x.size())
    print(y.size())

    # assert y.size() == x.size()
