import torch
import torch.nn as nn
import torch.optim as optim

class ProjectedGradientDescent(nn.Module):
    def __init__(self, lr=0.1, steps=10, device='cuda'):
        """
        Initialize the Projected Gradient Descent (PGD) module.

        Args:
        - lr: Learning rate for gradient descent.
        - steps: Number of PGD iterations.
        - device: The device to run the computation on ('cpu' or 'cuda').
        """
        super(ProjectedGradientDescent, self).__init__()
        self.lr = lr
        self.steps = steps
        self.device = device

    def project_to_density_matrix(self, matrix):
        """
        Projects a Hermitian matrix onto the space of valid density matrices:
        - Hermitian
        - PSD (Positive Semi-Definite)
        - Trace 1
        """
        # Make it Hermitian
        matrix = (matrix + matrix.conj().T) / 2

        # Eigen-decomposition (still differentiable in PyTorch)
        eigvals, eigvecs = torch.linalg.eigh(matrix)

        # Clamp eigenvalues to be >= 0 (enforce PSD)
        eigvals_clamped = torch.clamp(eigvals, min=0.0)

        # Reconstruct matrix
        projected = (eigvecs @ torch.diag(eigvals_clamped) @ eigvecs.conj().T)

        # Normalize trace to 1
        projected = projected / torch.trace(projected)

        return projected

    def forward(self, nn_densityMatrix):
        """
        Performs PGD to find the closest density matrix to nn_densityMatrix.
        This method will be called during inference to optimize the density matrix.
        
        Args:
        - nn_densityMatrix: Input non-physical matrix, typically output from a neural network.
        
        Returns:
        - Optimized density matrix
        """
        # Ensure nn_densityMatrix is on the correct device
        nn_densityMatrix = nn_densityMatrix.to(self.device)

        # Start with a copy of the input matrix
        rho = nn_densityMatrix.clone().detach().requires_grad_(True)

        # Use SGD for optimization
        optimizer = optim.SGD([rho], lr=self.lr)

        # Iteratively optimize with PGD
        for _ in range(self.steps):
            optimizer.zero_grad()

            # Loss: distance to original (can use Frobenius, L1, etc.)
            loss = torch.norm(rho - nn_densityMatrix, p='fro')
            loss.backward()

            # Gradient step
            optimizer.step()

            # Projection step to enforce density matrix constraints
            with torch.no_grad():
                rho.data = self.project_to_density_matrix(rho.data)

        return rho

