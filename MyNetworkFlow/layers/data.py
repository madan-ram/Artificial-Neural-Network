import numpy as np
import init
from base import Layer
from  utils import bcolors
from activation import get_activation
import sys, os

class Variable(Layer):

	def __init__(self, shape, name):
		self.shape = shape
		# super(Variable, self).__init__(None, np.prod(shape[1:]))
		# if isinstance(incoming, np.ndarray):
		# 	self.value = incoming
		# else:
		# 	print(bcolors.FAIL+'Variable can only take numpy, with name='+name+bcolors.ENDC)
		# 	sys.exit(0)

	def forward(self):
		pass

	def backward(self, delta=None, forward_delta=None):
		pass

	def update(self, updates, lr=0.1):
		for layer in updates:
			for key in layer.parm:
				layer.parm[key] = layer.parm[key] - lr * layer.diff[key]

	def get_batch_dim(self):
		return self.shape[0]

	def get_input_dim(self):
		return np.prod(self.shape[1:])

if __name__ == '__main__':

	num_batch = 32
	input_dim = 10
	input = np.random.random((num_batch, input_dim))
	incoming = Variable(input, 'test')
	if ((incoming.shape == (num_batch, input_dim)) 
		and (incoming.get_batch_dim() == 32)
		and (incoming.get_output_dim() == input_dim)):

		print('Variable'+' '+bcolors.OKGREEN+'PASS'+bcolors.ENDC)
