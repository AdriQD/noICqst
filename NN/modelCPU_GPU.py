import torch
import pandas as pd
import torch.nn as nn
import torch.jit
import cvxpy as cp
from qutip import fidelity, Qobj
from utils.projection-old-new import NewProjB



@torch.jit.script
def reconstruction(nn_out,norma):
	projected_dm= NewProjB(nn_out, norma, False)[1]
	
	return projected_dm

class gpu_cpu(nn.Module):
	def __init__(self, input_len, out_channel, nheads, dim_ff, foldertype, tiponorma ):
		super().__init__()

		kernel = 3
		kernel2 = 3
		

		padd = 1
		strid1 = 1
		strid2 = 1
		model_dim = int((input_len - kernel + 2*padd)/strid1)+1

		#correlators
		f = open('data/fromMeasurementSettings/'+foldertype+'/corrsCountList.txt', 'r')
		self.corrs_counts = f.read()
		self.norma = tiponorma

		#ENCODING
		self.conv1 = nn.Conv1d(1, out_channel, kernel_size = kernel, padding = padd,stride = strid1 )
		self.conv2 = nn.Conv1d(out_channel, 1, kernel_size = kernel2, padding = padd, stride = strid2 )
		

		#RECONSTRUCTION

		self.enc_transf = nn.TransformerEncoderLayer(d_model = model_dim, nhead = nheads, dim_feedforward = dim_ff, batch_first = True, norm_first=True)
		self.relu = torch.nn.ReLU()
		self.enc_stack = nn.TransformerEncoder(self.enc_transf, num_layers=1)
		self.T = torch.nn.Tanh()
		self.G = torch.nn.GELU()




	def forward(self, xIn,xTarget):

		#ENCODING

		x = self.T(self.conv1(xIn)) #this for pauli's
		#x= F.selu(self.conv1(x))

		#DECODING

		x= self.G(self.enc_stack(x))
		x = self.relu(self.conv2(x))
		x = torch.flatten(x,1) 
		
		# asyncronous CPU computation
		projectedNNdm = torch.jit.fork(reconstruction,x.cpu(), xTarget.cpu())
		xProj = torch.jit.wait(projectedNNdm)

		return torch.tensor(xProj.to("cuda")) #move back to GPU


