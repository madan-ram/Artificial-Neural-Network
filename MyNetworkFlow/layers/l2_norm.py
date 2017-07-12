import numpy as np
import init
from base import Layer
from data import Variable

class L2_norm(Layer):

	def __init__(self, label, incoming, name):
		super(L2_norm, self).__init__(label, incoming)
		self.label = label
		self.incoming = incoming
		self.name = name

	def forward(self):
		
		# when forward is called, we call previous layer forward
		self.incoming.forward()

		# Paramater to be share to the previous layer
		self.delta = self.incoming.value - self.label.value
		self.value = (1/2.)*np.mean(np.sum(np.power(self.delta, 2), axis=1))

	def backward(self, delta=None, forward_delta=None):
		self.diff = {self.name+'_delta': self.delta}

		self.incoming.backward(self.delta, None)

	def update(self):
		updates=[]
		self.incoming.update(updates)

if __name__ == '__main__':

	# Test above code if it works for numpy label
	num_batch = 32
	input_dim = 10

	incoming_numpy = np.random.random((num_batch, input_dim))
	incoming = Variable(incoming_numpy, 'incoming')

	label = np.random.random((num_batch, input_dim))

	loss = L2_norm(label, incoming, 'loss')

	loss.forward()
	print(loss.loss)
	loss.backward()
	print(loss.delta.shape)