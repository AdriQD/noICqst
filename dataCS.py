import numpy as np
import scipy as sp
import cvxpy as cp
import random

import configparser

config = configparser.ConfigParser()
config.read('config.ini')


NumQubits = config['DATA']['NumQubit']
NumShots = config['DATA']['NumShots']
NumPauliStrings = config['DATA']['NumPauliStrings']
path = config['DATA']['PauliFilePath']


import random

def generate_unique_base3_numbers(N, length,path):
	"""
	Generate a list of N unique random base-3 numbers (strings of digits 0, 1, 2).
	
	Parameters:
		N (int): Number of unique numbers to generate.
		length (int): Length of each number.
	
	Returns:
		List[str]: A list of N unique base-3 numbers as strings.
	
	Raises:
		ValueError: If N is greater than the number of possible unique base-3 numbers.
	"""
	max_possible = 3 ** length
	if N > max_possible:
		raise ValueError(f"Cannot generate {N} unique numbers with length {length} — max is {max_possible}")
	
	generated = set()
	while len(generated) < N:
		num = ''.join(random.choice('012') for _ in range(length))
		generated.add(num)
	
	np.save(path, generated, allow_pickle=True)


class CSdata():
	
	def __init__(self, NumQubits, NumShots, NumPauliStrings, path):

		self.nq = NumQubits
		
		self.NumSh = NumShots
		self.nps = NumPauliStrings
		self.path = path

	def string2paulibasis(self):
		"""
		Given a list of binary strings, gives the Pauli bases. 
		PARAMETERS:
			path: path to the list of strings of the form "012..." where  we do the following 0:X, 1: Y, 2:Z
		RETURNS:
			bases: (2**n_qubits, 2**n_qubits, len(idx)) complex array. It contains
				  the bases associated to the Pauli strings in idx.
		"""

		idx = np.load(self.path, allow_pickle=True)
		# define the one-qubit bases
		X = np.array([[1., 1.], [1., -1.]], dtype=complex)/np.sqrt(2)
		Y = np.array([[1, 1], [1.*1j, -1.*1j]], dtype=complex)/np.sqrt(2) 
		Z = np.array([[1, 0], [0, 1]], dtype=complex)
		sigma = np.stack([X, Y, Z], axis=-1)
		n_qubits = len(idx[0])
		bases = np.zeros((2**n_qubits, 2**n_qubits, len(idx)), dtype="complex")
		# calculate all the combinations of tensor products
		for i in range(len(idx)):
			basis_i = 1
			for k in range(n_qubits):
				basis_i = np.kron(basis_i, sigma[:, :, int(idx[i][k])])
			bases[:, :, i] = basis_i
		return bases


	def base2operator(self,matrix, vec=False):
		"""
		Dada una lista de bases construye la representacion vectorial de el POVM

		eng: from a collection of PualiString "010,222 etc" it switches the string elements with the associated POVM (Pauli)
		matrices
		input:
			matrix: array of dim x N. Bases previously 
			generated from the function string2paulibasis

		returns:
			vectors: array di x N. Returns a list of N observables
		"""
		dim = matrix.shape
		if len(dim) == 2:
			A = np.zeros((dim[0], dim[0], dim[1]), dtype="complex")
			for k in range(dim[1]):
				evector = matrix[:, k].reshape(-1, 1)
				projector = np.kron(evector.conj().T, evector)
				A[:, :, k] = projector
		else:
			A = np.zeros((dim[0], dim[0], dim[2]*dim[1]), dtype="complex")
			for j in range(dim[2]):
				for k in range(dim[1]):
					evector = matrix[:, k, j].reshape(-1, 1)
					projector = np.kron(evector.conj().T, evector)
					A[:, :, dim[0]*j + k] = projector/int(matrix.shape[-1])
		if vec:
			A = A.reshape(dim[0]**2, -1, order="F")

		return A


	def density_matrix(self):
		
		"""
		Random density matrix of dimension di sampled form Haar measure
		"""
		dim = 2**self.nq
		psi = sp.stats.unitary_group.rvs(dim)[:, 0]
		dens = np.outer(psi.conj(), psi)

		return dens


	def compressed_sensing(self, prob, A, epsilon):
		"""
		Compressed sensing reconstruction (finally).

		Gets the density matrix using the matrix of POVMs A and the measured
		probabilities using compressive sensing.

		inputs:
			prob: array (di*N, ). probabilities
			A: array (di, di, N). Matrix with all the POVM elements
			epsilon: error tollerance 
		returns:
			dens: array (di, di). density matrix.
		"""

		# optimizacion convexa
		di2 = A.shape[0]
		di = int(np.sqrt(di2))
		de = cp.Variable((di, di), hermitian=True)
		constraints = [de >> 0]
		constraints += [cp.trace(de) == 1]
		constraints += [cp.norm(A.conj().T@cp.reshape(de, (di2, 1), order="F")
					   - prob.reshape(-1, 1)) <= epsilon]
		objective = cp.Minimize(cp.normNuc(de))
		problema = cp.Problem(objective, constraints)
		result = problema.solve()

		return de.value/np.trace(de.value)


	def GenerateData(self , dens , wantprobes : False):
	
			#dens = self.density_matrix(dim)
			paulis = self.string2paulibasis(self.path)
			# vectorized version of the POVM
			vec_paulis = self.base2operator(paulis, True)
			probs = np.abs(vec_paulis.conj().T@dens.reshape(-1, 1, order="F"))
			exp_probs = np.random.multinomial(self.NumSh, probs.flatten())/self.NumSh

			#approximation of CS tolerance for a given number of shots, as per article ....
			tolerance = np.sqrt(np.sum(exp_probs*(1 - exp_probs))/self.NumSh)

			DM_withShots = self.compressed_sensing(exp_probs, vec_paulis, tolerance)

			if wantprobes:
				return (dict(theoreticalProb = probs, experimentalProb =  exp_probs, epsilon = tolerance ))
			else:
				return dict(csdm = DM_withShots ,thdm = dens)

	def ParallelGenData(self , DatasetDim,  wantprobes : False):
	
			dens = self.density_matrix(self.nq)
			paulis = self.string2paulibasis(self.path)
			# vectorized version of the POVM
			vec_paulis = self.base2operator(paulis, True)
			probs = np.abs(vec_paulis.conj().T@dens.reshape(-1, 1, order="F"))
			exp_probs = np.random.multinomial(self.NumSh, probs.flatten())/self.NumSh

			#approximation of CS tolerance for a given number of shots, as per article ....
			tolerance = np.sqrt(np.sum(exp_probs*(1 - exp_probs))/self.NumSh)

			DM_withShots = self.compressed_sensing(exp_probs, vec_paulis, tolerance)

			if wantprobes:
				return (dict(theoreticalProb = probs, experimentalProb =  exp_probs, epsilon = tolerance ))
			else:
				return dict(csdm = DM_withShots ,thdm = dens)

