import torch
import torch.nn as nn
from mamba_ssm import Mamba
from einops import rearrange

class Bi_mamba_block(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout=0.2, device="cuda"):
        super().__init__()
        self.device = device

        self.forward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        ).to(device)

        self.backward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        ).to(device)

        self.norm1 = nn.LayerNorm(d_model).to(device)
        self.norm2 = nn.LayerNorm(d_model).to(device)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        ).to(device)
    
    def forward(self, x):

        residual = x
        x_norm = self.norm1(x)
        # lstm_out, _ = self.lstm(x)
        forward_mamba = self.forward_mamba(x_norm)

        backward_out = self.backward_mamba(torch.flip(x_norm, dims=[1]))
        backward_out = torch.flip(backward_out, dims=[1])

        mamba_out = forward_mamba + backward_out
        mamba_out = self.norm2(mamba_out)
        ff_out = self.feed_forward(mamba_out)
        output = ff_out + residual
        # dense_out = self.dense(output)
        
        
        return output  # No activation here ! It is done by the cross entropy loss


class segmentation_model(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout=0.2, device="cuda"):
        super().__init__()
        self.device = device
        n_bimamba = 5
        self.Bi_blocks = nn.Sequential(*[Bi_mamba_block(d_model, d_state, d_conv, expand) for i in range(n_bimamba)])      
        self.dense = nn.Linear(d_model, 3).to(device)

    def forward(self, x):

        out_blocks = self.Bi_blocks(x)
        dense_out = self.dense(out_blocks)       
        return dense_out  # No activation here ! It is done by the cross entropy loss


class K_regression(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout=0.3, device="cuda"):
        super().__init__()
        self.device = device
        n_bimamba = 2
        self.Bi_blocks = nn.Sequential(*[Bi_mamba_block(d_model, d_state, d_conv, expand) for i in range(n_bimamba)])      

        self.dense = nn.Linear(in_features=199 * d_model, out_features=2).to(device)

        self.relu = nn.ReLU()

    def forward(self, x):

        mamba_out = self.Bi_blocks(x)

        mamba_out = rearrange(mamba_out, "b l c -> b (l c)")
        out = self.dense(mamba_out)
        out = self.relu(out)
        out = torch.clamp(out, min=0, max=1e12)
        out[out < 1e-7] = 0
        return out


class alpha_regression(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout=0.3, device="cuda"):
        super().__init__()
        self.device = device
        n_bimamba = 2
        self.Bi_blocks = nn.Sequential(*[Bi_mamba_block(d_model, d_state, d_conv, expand) for i in range(n_bimamba)])      

        self.dense = nn.Linear(in_features=199 * d_model, out_features=2).to(device)

        self.softplus = nn.Softplus()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        mamba_out = self.Bi_blocks(x)
        mamba_out = rearrange(mamba_out, "b l c -> b (l c)")

        out = self.dense(mamba_out)
        out = self.sigmoid(out)*2
        
        return out

class all_at_the_same_time(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout=0.2, device="cuda"):
        super().__init__()
        self.device = device

        self.model_K = K_regression(
            d_model+1, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout
        )
        self.model_alpha = alpha_regression(
            d_model+1, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout
        )
        self.segmentation = segmentation_model(
            d_model, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout
        )

    def forward(self, x):
        probas = self.segmentation(x)

        classes = torch.argmax(torch.softmax(probas, dim=2), dim=2)

        classes[x[:, :, 0] == 0] = 0

        classes[x[:, :, 0] == 0] = 0
        classes = classes.unsqueeze(-1)  # adding a dimension for the next step
        concat_entry = torch.cat((classes, x[:,:,:]), dim=2)

        alpha = self.model_alpha(concat_entry)

        K = self.model_K(concat_entry)

        return probas, alpha, K