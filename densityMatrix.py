import numpy as np
import scipy as sp
from multiprocessing import Pool

# Define the DensityMatrixIterator class
class DensityMatrixIterator:
	def __init__(self, nq):
		self.nq = nq
		self.dim = 2**self.nq

	def __iter__(self):
		return self  # The iterator object itself is returned

	def __next__(self):
		# Generate a random unitary matrix from Haar measure
		psi = sp.stats.unitary_group.rvs(self.dim)[:, 0]
		# Create the density matrix
		dens = np.outer(psi.conj(), psi)
		return dens

# Define a function to be used by Pool.map, which calls __next__()
def generate_density_matrix(iterator):
	return next(iterator)



if __name__ == "__main__":
	
	# Number of qubits (nq) and the number of random density matrices to generate
	nq = 3
	num_matrices = 5
	# Create a list of DensityMatrixIterator instances, one for each matrix to generate
	iterators = [DensityMatrixIterator(nq) for _ in range(num_matrices)]

	# Create a Pool of worker processes
	with Pool() as pool:
		# Map the generate_density_matrix function to each iterator in the list
		result = pool.map(generate_density_matrix, iterators)

	# Print the result
	for i, dens in enumerate(result):
		print(f"Density matrix {i + 1}:\n", dens)
