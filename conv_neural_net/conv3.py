from scipy import signal
import numpy as np
import time
import cython as cy

class ConvolutionNeuralNetwork:
	def setWeights(self, W = None):
		if W == None:
			self.W = np.array(
			[
				[[ 1,  1],
				[-1, -1]],
				[[-1, -1],
				[ 1,  1]],
				[[ 1, -1],
				[ 1, -1]],
				[[-1,  1],
				[-1,  1]],
				[[-1,  1],
				[ 1, -1]],
				[[ 1, -1],
				[-1,  1]],
				[[ 0,  1],
				[ 1,  0]]
			]
			)
		else:
			self.W = W

	def conv(self, old_map):
		new_map = []
		for i, j in np.ndindex(old_map.shape[0], self.W.shape[0]):
			grad = signal.convolve2d(old_map[i], self.W[j], boundary='symm', mode='same')
			new_map.append(grad)
		return np.asarray(new_map)

	def max_pooling(self, old_map, size=(2, 2)):
		if old_map.shape[1] % size[0] == 0:
			it_0 = old_map.shape[1] / size[0]
		else:
			it_0 = (old_map.shape[1] / size[0]) + 1

		if old_map.shape[2] % size[1] == 0:
			it_1 = old_map.shape[2] / size[1]
		else:
			it_1 = (old_map.shape[2] / size[1]) + 1

		new_map = np.zeros(( old_map.shape[0], it_0, it_1))
		count = 0
		for t, i, j in  np.ndindex((old_map.shape[0], it_0, it_1)):
			new_map[t, i, j] = np.max(old_map[t, i * size[0]:(i+1) * size[0], j * size[1]:(j+1) * size[1]])
		return new_map

L1 = ConvolutionNeuralNetwork()
L1.setWeights()


X = np.arange(400).reshape(4, 10, 10)

s = time.time()
print L1.conv(X).shape
print time.time() - s

s = time.time()
print L1.max_pooling(X, size=(4, 4)).shape
print time.time() - s
