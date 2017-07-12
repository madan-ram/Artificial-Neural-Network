import numpy as np

class Layer(object):

	def __init__(self, incoming, num_uints):
		self.incoming = incoming
		self.num_uints = num_uints
		self.shape = (self.get_batch_dim(), self.num_uints)

	def get_batch_dim(self):
		return self.incoming.shape[0]

	def get_input_dim(self):
		return np.prod(self.incoming.shape[1:])

	def get_output_dim(self):
		return self.num_uints

