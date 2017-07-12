import numpy as np
from  utils import bcolors

class Sigmoid:

	def forward(self, x):
		self.sigmoid = 1.0/(1+np.exp(-x))
		return self.sigmoid

	def backward(self):
		return self.sigmoid*(1-self.sigmoid)

class Tanh:

	def forward(self, x):
		self.tanh = np.tanh(x)
		return self.tanh

	def backward(self):
		return 1.0 - np.power(self.tanh, 2)

class Linear:

	def forward(self, x):
		self.linear = x
		return self.linear

	def backward(self):
		return np.ones(self.linear.shape)

def get_activation(activation):
	
	activation = str.lower(activation)

	if activation == 'linear':
		return Linear()
	elif activation == 'tanh':
		return Tanh()
	elif activation == 'sigmoid':
		return Sigmoid()
	else:
		print(bcolors.FAIL+"'activation' to get_activation, not found, using linear"+bcolors.ENDC)
		return Linear()


def euclidean(x, y):
	t = (x-y)
	return np.sqrt(np.sum(t**2))
		
if __name__ == '__main__':

	def computeNumericalGradient(O, x, eps = 10**-6):
		perturb = np.zeros(x.shape)
		return ((O.forward(x+eps) - O.forward(x-eps))/(2.*eps))

	x = np.random.uniform(-1, 1, (1, 10))
	for f in ['tanh', 'linear', 'sigmoid']:
		O = get_activation(f)
		eps = 10**-6
		if euclidean(computeNumericalGradient(O, x, eps), O.backward()) < 10**-5:
			print(f+' '+bcolors.OKGREEN+'PASS'+bcolors.ENDC)
