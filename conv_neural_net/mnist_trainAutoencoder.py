import numpy as np
import theano
import theano.tensor as T
import cPickle as pickle
import gzip, numpy
import AutoEncoder as ae
import cv2
from utils import tile_raster_images


def test(Weights, counter, ext, channel=1):
	"""this is an utility that takes weights and plot there feature as image"""
	tile_shape = (15, 20)
	image_resize_shape = (2, 2)
	if channel == 1:
		img = tile_raster_images(X=Weights.T, img_shape=(window_size, window_size), tile_shape=tile_shape, tile_spacing=(1, 1))
		newimg = np.zeros((img.shape[0]*image_resize_shape[0], img.shape[1]*image_resize_shape[1]))
		for i in xrange(img.shape[0]):
			for j in xrange(img.shape[1]):
				newimg[i*image_resize_shape[0]:(i+1)*image_resize_shape[0], j*image_resize_shape[1]:(j+1)*image_resize_shape[1]] = img[i][j] * np.ones(image_resize_shape)
		cv2.imwrite('tmp/'+str(counter)+'_'+ext+'.jpg', newimg)
	else:
		tile = Weights.shape[0] / channel
		i = 0
		temp = (Weights.T[:, tile*i:(i+1)*tile], Weights.T[:, (i+1)*tile:(i+2)*tile], Weights.T[:, (i+2)*tile:tile*(i+3)])
		img = tile_raster_images(X=temp, img_shape=(window_size, window_size), tile_shape=tile_shape, tile_spacing=(1, 1))
		newimg = cv2.resize(img, (img.shape[0] * image_resize_shape[0],img.shape[1] * image_resize_shape[1]))
		cv2.imwrite('tmp/'+str(counter)+'_'+ext+'.jpg', newimg)

window_size = 28
training_epochs = 50
batch_size = 20

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f)

aeL1 = ae.AutoEncoder(number_of_input_layer=window_size*window_size, number_of_hidden_layer=300)
trainL1 = aeL1.fit(corruption_quantity=0.0, learning_rate=0.00005, batch_size=batch_size, have_sparsity_penalty=True, cost_type='SQUAREMEAN', learning_type='RMSPROP')

X, y = train_set[0], train_set[1]

aeL1.add_sample(X)

n_train_batches = X.shape[0]/ batch_size
for epoch in xrange(training_epochs):
	c = []
	for batch_index in xrange(n_train_batches):
		c.append(trainL1(batch_index))
	fwWeights = open('mnist.pickle_W1_'+str(epoch), 'w')
	W1 = aeL1.W.get_value()
	print >> fwWeights, pickle.dumps(W1)
	fwWeights.close()
	print len(c), np.mean(c)
	test(aeL1.W.get_value(), epoch, 'mnist__W1__', channel=1)