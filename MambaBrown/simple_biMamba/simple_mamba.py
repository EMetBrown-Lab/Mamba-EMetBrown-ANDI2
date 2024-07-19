import torch
import torch.nn as nn
from mamba_ssm import Mamba
from einops import rearrange

class segmentation_model(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout=0.2, device="cuda"):
        super().__init__()
        self.device = device

        self.mamba = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        ).to(device)

        self.flipped_mamba = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        ).to(device)

        self.dropout = nn.Dropout(p=dropout).to(device)
        self.fc = nn.Linear(in_features=d_model, out_features=3).to(device)
        self.softplus = nn.Softplus()

    def forward(self, input):
        

        mamba_out = self.mamba(input)
        mamba_flipped_out = self.flipped_mamba(torch.flip(input, dims=[1]))
        mamba_out = mamba_out + mamba_flipped_out

        mamba_out = self.dropout(mamba_out)
        out = self.fc(mamba_out)

        return out  # No activation here ! It is done by the cross entropy loss


class K_regression(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout=0.3, device="cuda"):
        super().__init__()
        self.device = device

        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        ).to(device)
        self.flipped_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        ).to(device)

        self.dropout = nn.Dropout(p=dropout).to(device)
        self.fc = nn.Linear(in_features=199 * d_model, out_features=2).to(device)

        self.softplus = nn.Softplus()

    def forward(self, input):

        mamba_out = self.mamba(input)
        mamba_flipped_out = self.flipped_mamba(torch.flip(input, dims=[1]))

        mamba_out = mamba_out + mamba_flipped_out

        mamba_out = rearrange(mamba_out, "b l c -> b (l c)")

        mamba_out = self.dropout(mamba_out)
        out = self.fc(mamba_out)
        out = torch.clamp(out, min=0, max=1e12)
        out[out < 1e-7] = 0
        return out


class alpha_regression(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout=0.3, device="cuda"):
        super().__init__()
        self.device = device

        self.mamba = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        ).to(device)
        self.flipped_mamba = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        ).to(device)

        self.dropout = nn.Dropout(p=dropout).to(device)
        self.fc = nn.Linear(in_features=199 * d_model, out_features=2).to(device)

        self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU()
    def forward(self, input):

        mamba_out = self.mamba(input)
        mamba_flipped_out = self.flipped_mamba(torch.flip(input, dims=[1]))

        mamba_out = mamba_out + mamba_flipped_out

        mamba_out = rearrange(mamba_out, "b l c -> b (l c)")

        mamba_out = self.dropout(mamba_out)
        out = self.fc(mamba_out)

        
        return self.sigmoid(out) * 2

class all_at_the_same_time(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout=0.2, device="cuda"):
        super().__init__()
        self.device = device

        self.model_K = K_regression(
            d_model, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout
        )
        self.model_alpha = alpha_regression(
            d_model, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout
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
        concat_entry = torch.cat((classes, x[:,:,:-1]), dim=2)

        alpha = self.model_alpha(concat_entry)

        K = self.model_K(concat_entry)

        return probas, alpha, K