import torch
from modelCPU_GPU import gpu_cpu

model = gpu_cpu

##here below a snippet of the inference step for the mixed model.

model.eval()
with torch.no_grad():
    latent_x = model.encode(xIn)  # access intermediate "x"
    output = model.forward(xIn)  # full forward with projection
