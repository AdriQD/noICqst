import torch
import pandas as pd
import torch.nn as nn
import torch.jit
import cvxpy as cp
from qutip import fidelity, Qobj
from utils.projection-old-new import NewProj



@torch.jit.script
def reconstruction(nn_out,norma):
	projected_dm= NewProj(nn_out, norma, False)[1]
	
	return projected_dm

import torch
import torch.nn as nn
import torch.nn.functional as F

class gpu_cpu(nn.Module):
    def __init__(self, input_len, out_channel, nheads, dim_ff, foldertype, tiponorma):
        super().__init__()

        kernel = 3
        padd = 1
        strid1 = 1
        strid2 = 1
        model_dim = int((input_len - kernel + 2 * padd) / strid1) + 1

        # Load measurement correlators
        with open(f'data/fromMeasurementSettings/{foldertype}/corrsCountList.txt', 'r') as f:
            self.corrs_counts = f.read()
        self.norma = tiponorma

        # Encoding layers
        self.conv1 = nn.Conv1d(1, out_channel, kernel_size=kernel, padding=padd, stride=strid1)
        self.conv2 = nn.Conv1d(out_channel, 1, kernel_size=kernel, padding=padd, stride=strid2)

        # Transformer encoder
        self.enc_transf = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nheads, dim_feedforward=dim_ff,
            batch_first=True, norm_first=True
        )
        self.enc_stack = nn.TransformerEncoder(self.enc_transf, num_layers=1)

        # Activations
        self.T = torch.nn.Tanh()
        self.G = torch.nn.GELU()
        self.relu = torch.nn.ReLU()

    def encode(self, xIn):
        """Returns the intermediate representation before projection."""
        x = self.T(self.conv1(xIn))                # First conv + Tanh
        x = self.G(self.enc_stack(x))              # Transformer + GELU
        x = self.relu(self.conv2(x))               # Second conv + ReLU
        x = torch.flatten(x, 1)                    # Flatten for projection
        return x

    def project(self, x_latent):
        """Performs the async CPU projection."""
        projectedNNdm = torch.jit.fork(reconstruction, x_latent.cpu())
        xProj = torch.jit.wait(projectedNNdm)
        return torch.tensor(xProj.to("cuda"))

    def forward(self, xIn):
        x_latent = self.encode(xIn)
        xProj = self.project(x_latent)
        return xProj
