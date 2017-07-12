import numpy as np
import init
from base import Layer
from data import Variable
from activation import get_activation

class Dense(Layer):

	def __init__(self, incoming, num_uints, name, activation='linear'):
		super(Dense, self).__init__(incoming, num_uints)
		self.name = name
		self.num_uints = num_uints
		self.activation = activation

		self.diff = {}
		self.parm = {}

		self.W = init.GlorotUniform()((self.get_input_dim(), self.get_output_dim()))
		self.b = init.Constant(0.01)((self.get_output_dim()))

		self.parm = {name+'_W': self.W,  name+'_b': self.b}

	def forward(self):

		# when forward is called, we call previous layer forward
		self.incoming.forward()

		self.activation_object = get_activation(self.activation)
		if isinstance(self.incoming, np.ndarray):
			self.value = self.activation_object.forward(np.dot(self.incoming, self.parm[self.name+'_W'])+self.parm[self.name+'_b'])
		else:
			self.value = self.activation_object.forward(np.dot(self.incoming.value, self.parm[self.name+'_W'])+self.parm[self.name+'_b'])

	def backward(self, prev_delta, prev_forward_delta=None):

		if prev_forward_delta is None:
			prev_forward_delta = np.asarray(1)
		

		self.delta = np.dot(prev_delta, prev_forward_delta.T)*self.activation_object.backward()
		
		self.forward_delta = self.parm[self.name+'_W']

		self.grad_b = self.delta
		self.grad_W = np.dot(self.incoming.value.T, self.delta)

		self.diff = {self.name+'_W': self.grad_W, self.name+'_b': self.grad_b}

		# # Call previous layer backward pass
		self.incoming.backward(self.delta, self.forward_delta)

	def update(self, updates):
		updates.append(self)
		self.incoming.update(updates)
		

if __name__ == '__main__':

	num_batch = 32
	input_dim = 1024
	input = np.random.random((num_batch, input_dim))
	incoming = Variable(input, 'test_var')

	full1 = Dense(incoming, 10, 'test_full')

	full1.forward()
	full1.backward(np.random.random((32, 10)))
