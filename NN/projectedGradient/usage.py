# Instantiate the module with desired learning rate and steps
pgd = ProjectedGradientDescent(lr=0.1, steps=20, device='cuda')  # Or 'cpu'

# Move model to device (GPU/CPU)
pgd.to('cuda')  # if you're using a GPU

# Assume nn_output is the output from your neural network
nn_output = torch.randn(1, 64, 64).cuda()  # Example: a batch of size 1 with a 64x64 matrix

# Get the optimized density matrix via PGD
optimized_matrix = pgd(nn_output)

# Now, optimized_matrix is a valid density matrix (Hermitian, PSD, Trace=1)
class gpu_cpu(nn.Module):
    def __init__(self, input_len, out_channel, nheads, dim_ff, foldertype, tiponorma, device='cuda'):
        super(gpu_cpu, self).__init__()
        self.device = device
        self.pgd = ProjectedGradientDescent(lr=0.1, steps=20, device=self.device)

    def forward(self, xIn, xTarget=None):
        # Run through the network's layers
        x_latent = self.encode(xIn)  # Assuming this is already implemented

        # Use PGD to project to valid density matrix
        xProj = self.pgd(x_latent)

        return xProj
