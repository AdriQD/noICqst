
import cvxpy as cp
from numpy import sqrt


#OLD PROJECTION

def proj(corrs_counts, corr_vec, corr_org, tiponorma,pauli_basis_4q):    

	variable_rho = cp.Variable((16,16), hermitian=True) #four qubit dm
	
	targ_mat=stato_from_corr(corr_vec,pauli_basis_4q) #for F
	#print(targ_mat)
	org_mat=stato_from_corr(corr_org,pauli_basis_4q) #for the constraints
	
	print(cp.installed_solvers())

	F=cp.norm(variable_rho-targ_mat, tiponorma)  
	
	# Set the remaining constraints
	constraints = [cp.trace(variable_rho) == 1, variable_rho >> 0] 
	
	# Construct the problem.
	objective = cp.Minimize(F)    
	
	###impose corr constarints
	for i in range(1,256): #not imposing normalization
		if corrs_counts[i-1]=="1":   
			corr_value=cp.real(cp.trace(pauli_basis_4q[i]@org_mat))
			constraints.append(cp.real(cp.trace(variable_rho@pauli_basis_4q[i])) == corr_value)   

	prob = cp.Problem(objective, constraints)
	result = [prob.solve(verbose=True,solver = 'SCS'), variable_rho.value]

	return result

# tentative new projection function that internally consideres density matrices
#I don't undesrstand why here the minimization is run against a random hermitian, instead of reducing the 
#difference between target and a-given matrix


def NewProj(nn_densityMatrix,  tiponorma):  

	"""
	Find the closer densty matrix given a neural network output. 

	inputs.
		nn_densityMatrix: the output from the network. its a set of matrix elemens
						We can force Trace =1 but ot positivity
		
	outputs.
		A =  the closer density matrix in the feasible set.
	"""

	# Dimensions
	di2 = nn_densityMatrix.shape[0]
	di = int(sqrt(di2))
	
	# Define the variable to optimize: a density matrix
	guess = cp.Variable((di, di), hermitian=True)
	target = cp.Constant(nn_densityMatrix)
	
	# Objective: make de close to nn_densityMatrix
	F = cp.norm(guess- target, tiponorma)  
	
	# Constraints
	constraints = [
	    guess >> 0,                           # Positivity
	    cp.trace(guess) == 1                # Unit trace
	]

	
	# Optimization problem
	objective = cp.Minimize(F)    
	proj_cvx = cp.Problem(objective, constraints)
	result_status = proj_cvx.solve(verbose=True, solver='SCS')
	
	optimized_matrix = guess.value
	return result_status, optimized_matrix


def NewProjB(nn_densityMatrix, tiponorma: str, verb: str):

	"""
	This function find the closest density matrix to a given random density matrix.

	input:
		nn_densityMatrix: this is a random density matrix input. In our case, the network output
		tiponorma: which type of norm we use for the optimization problem.
		verb: True or False, verbose.

	returns:
		projectionDM : the closest density matrix, according to the norm provided.

	"""

	# Define the optimization variable X (the matrix we want to find)
	X = cp.Variable(nn_densityMatrix.shape, complex=True)

	# Define the constraints
	constraints = [
		cp.Hermitian(X),            # X must be Hermitian
		X >> 0,                     # X must be positive semidefinite
		cp.trace(X) == 1            # Trace of X must be 1
	]

	# Define the objective function (minimize Frobenius norm)
	objective = cp.Minimize(cp.norm(nn_densityMatrix - X, tiponorma))

	# Set up and solve the problem
	problem = cp.Problem(objective, constraints)
	problem.solve(verbose=verb, solver='SCS')

	# The result will be the closest density matrix
	projectionDM = X.value
	return projectionDM
	

