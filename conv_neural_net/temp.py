import sae as sparse_autoencoder
import sys
import os
from os import listdir
# import theano
import numpy as np
from os.path import isfile, join
import cv2
from utils import tile_raster_images
import pickle
import random
import time
import scipy.optimize
import scipy

def getFiles(dir_path):
	"""getFiles : gets the file in specified directory

	dir_path: String type
	dir_path: directory path where we get all files
	"""
	onlyfiles = [ f for f in listdir(dir_path) if isfile(join(dir_path, f)) ]
	return onlyfiles

def getImmediateSubdirectories(dir):
	"""
		this function return the immediate subdirectory list
		eg:
			dir
				/subdirectory1
				/subdirectory2
				.
				.
		return ['subdirectory1',subdirectory2',...]
	"""
	return [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]

# def create_image_patches(images, patch_size, stride=1):
# 	image_patches = []
# 	for img in images:
# 		for i in xrange(0, img.shape[0] - patch_size[0], stride):
# 			for j in xrange(0, img.shape[1] - patch_size[1], stride):
# 				temp = []
# 				for k in xrange(0, img.shape[2]):
# 					temp.append(img[i:i+patch_size[0], j:j+patch_size[1], k].ravel())
# 				image_patches.append(np.concatenate(temp))
# 	return np.asarray(image_patches, dtype=theano.config.floatX)

def test(Weights, counter, ext, channel=1):
	"""this is an utility that takes weights and plot there feature as image"""
	tile_shape = (8, 8)
	image_resize_shape = (5, 5)
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

window_size = 7
training_epochs = 50
batch_size = 40
file_batch_size = 50
stride = 3
fimgList = getFiles(sys.argv[1])
random.shuffle(fimgList)
lambda_ = 3e-3

number_of_file_batch = len(fimgList) / file_batch_size

sae1_theta = None#sparse_autoencoder.initialize(64, window_size*window_size*3)
for epoch in xrange(training_epochs):
	for batch_index_file in xrange(number_of_file_batch):
		image_patchs = []
		for fimg in fimgList[batch_index_file * file_batch_size:(batch_index_file+1) * file_batch_size]:
			img = cv2.imread(sys.argv[1]+'/'+fimg)
			img = cv2.resize(img, (200,(200*img.shape[1])/img.shape[0])) / 255.0

			for i in xrange(0, img.shape[0] - window_size, stride):
				for j in xrange(0, img.shape[1] - window_size, stride):
					image_patchs.append(np.concatenate((img[i:i+window_size, j:j+window_size, 0].ravel(), img[i:i+window_size, j:j+window_size, 1].ravel(), img[i:i+window_size, j:j+window_size, 2].ravel())))
		data = np.asarray(image_patchs)
		J = lambda x: sparse_autoencoder.sparse_autoencoder_cost(x, data.shape[1], 64,
	                                                         lambda_, 0.1,
	                                                         3, data.T)
		options_ = {'maxiter': 8, 'disp': True}
		if sae1_theta == None:
			sae1_theta = sparse_autoencoder.initialize(64, window_size*window_size*3)
		else:
			sae1_theta = result.x
		result = scipy.optimize.minimize(J, sae1_theta, method='L-BFGS-B', jac=True, options=options_)
	
	W1 = (result.x[0:64 * 147]).reshape((64, 147)).T
	print W1.shape
	test(W1, epoch, 'W1', channel=3)
# sae1_opt_theta = result.x

# print sae1_opt_theta.shape


# #-----------------------------------------------------------------------------------------------------------------------------------
# mean_mask = None

# aeL1 = AutoEncoder.AutoEncoder(number_of_inputLayer=window_size*window_size*3, number_of_hiddenLayer=64)
# #trainL1 = aeL1.fit(corruption_quantity=0.30, learning_rate=1, batch_size=batch_size, have_sparsity_penalty=False, output_type_non_linearity='RELU', type_cost='SQUAREMEAN', learning_type='RMSPROP')
# trainL1 = aeL1.fit(corruption_quantity=0.0, learning_rate=0.00005, batch_size=batch_size, have_sparsity_penalty=True, type_cost='CROSSENTROPY', learning_type='RMSPROP')

# for epoch in xrange(training_epochs):
# 	st = time.time()
# 	c = []
# 	print 'for epoch ', epoch
# 	for batch_index_file in xrange(number_of_file_batch):
# 		image_patchs = []
# 		print '	reading data in range(', batch_index_file * file_batch_size, ',', (batch_index_file+1) * file_batch_size, ')'
# 		for fimg in fimgList[batch_index_file * file_batch_size:(batch_index_file+1) * file_batch_size]:

# 			img = cv2.imread(sys.argv[1]+'/'+fimg)
# 			img = cv2.resize(img, (200,(200*img.shape[1])/img.shape[0])) / 255.0
			
# 			for i in xrange(0, img.shape[0] - window_size, stride):
# 				for j in xrange(0, img.shape[1] - window_size, stride):
# 					image_patchs.append(np.concatenate((img[i:i+window_size, j:j+window_size, 0].ravel(), img[i:i+window_size, j:j+window_size, 1].ravel(), img[i:i+window_size, j:j+window_size, 2].ravel())))

# 		data = np.asarray(image_patchs, dtype=theano.config.floatX)

# 		# if mean_mask == None:
# 		# 	mean_mask = np.mean(data, axis=0)
# 		# data = (data - mean_mask)

# 		aeL1.add_sample(data)
# 		if data.shape[0] % batch_size == 0:
# 			n_train_batches = data.shape[0]/batch_size
# 		else:
# 			n_train_batches = (data.shape[0]/batch_size) + 1
# 		for batch_index in xrange(n_train_batches):
# 			c.append(trainL1(batch_index))
# 	print 'Training epoch %d, cost ' % epoch, np.mean(c), ' time consumed ', time.time() - st
# 	test(aeL1.W.get_value(), epoch, 'W1', channel=3)
# 	fwWeights = open('Weights.pickle_W1_'+str(epoch), 'w')
# 	W1 = aeL1.W.get_value()
# 	print >> fwWeights, pickle.dumps(W1)
# 	fwWeights.close()





# window_size = 28
# training_epochs = 50
# batch_size = 20
# file_batch_size = 400
# stride = window_size
# fimgList = getFiles(sys.argv[1])
# random.shuffle(fimgList)

# number_of_file_batch = len(fimgList) / file_batch_size


# #-----------------------------------------------------------------------------------------------------------------------------------

# aeL1 = AutoEncoder.AutoEncoder(number_of_inputLayer=window_size*window_size, number_of_hiddenLayer=64)
# trainL1 = aeL1.fit(corruption_quantity=0.30, learning_rate=0.5, batch_size=batch_size, have_sparsity_penalty=False, type_cost='SQUAREMEAN')

# import cPickle, gzip, numpy

# f = gzip.open('mnist.pkl.gz', 'rb')
# train_set, valid_set, test_set = cPickle.load(f)

# mnist = train_set[0]

# print mnist.shape

# aeL1.add_sample(mnist)

# n_train_batches = mnist.shape[0]/ batch_size
# for epoch in xrange(50):
# 	c = []
# 	for batch_index in xrange(n_train_batches):
# 		c.append(trainL1(batch_index))
# 	test(aeL1.W.get_value(), epoch, 'W1', channel=1)
# 	print len(c), np.mean(c)
