import AutoEncoder
import sys
import os
from os import listdir
import theano
import numpy as np
from os.path import isfile, join
import cv2
from utils import tile_raster_images
import pickle
import random
import time
from ConvImage import Conv

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

def create_image_patches(images, patch_size, stride=1):
	image_patches = []
	for img in images:
		for i in xrange(0, img.shape[0] - patch_size[0], stride):
			for j in xrange(0, img.shape[1] - patch_size[1], stride):
				temp = []
				for k in xrange(0, img.shape[2]):
					temp.append(img[i:i+patch_size[0], j:j+patch_size[1], k].ravel())
				image_patches.append(np.concatenate(temp))
	return np.asarray(image_patches, dtype=theano.config.floatX)

def test(Weights, counter, ext, channel=1):
	"""this is an utility that takes weights and plot there feature as image"""
	tile_shape = (8, 8)
	image_resize_shape = (10, 10)
	img_shape = (window_size, window_size)
	newimg = None
	if channel == 1:
		img = tile_raster_images(X=Weights.T, img_shape=img_shape, tile_shape=tile_shape, tile_spacing=(1, 1))
		newimg = np.zeros((img.shape[0]*image_resize_shape[0], img.shape[1]*image_resize_shape[1]))
		for i in xrange(img.shape[0]):
			for j in xrange(img.shape[1]):
				newimg[i*image_resize_shape[0]:(i+1)*image_resize_shape[0], j*image_resize_shape[1]:(j+1)*image_resize_shape[1]] = img[i][j] * np.ones(image_resize_shape)
		cv2.imwrite('tmp/'+str(counter)+'_'+ext+'.jpg', newimg)
	elif channel == 3:
		tile = Weights.shape[0] / channel
		i = 0
		temp = (Weights.T[:, tile*i:(i+1)*tile], Weights.T[:, (i+1)*tile:(i+2)*tile], Weights.T[:, (i+2)*tile:tile*(i+3)])
		img = tile_raster_images(X=temp, img_shape=img_shape, tile_shape=tile_shape, tile_spacing=(1, 1))
		newimg = cv2.resize(img, (img.shape[0] * image_resize_shape[0],img.shape[1] * image_resize_shape[1]))
		cv2.imwrite('tmp/'+str(counter)+'_'+ext+'.jpg', newimg)
	else:
		temp = []
		Weights = Weights.reshape((window_size*window_size, 64, 64))
		for k in xrange(Weights.shape[1]):
			img = tile_raster_images(X=Weights[:,k, :].T, img_shape=img_shape, tile_shape=tile_shape, tile_spacing=(1, 1))
			newimg = np.zeros((img.shape[0]*image_resize_shape[0], img.shape[1]*image_resize_shape[1]))
			for i in xrange(img.shape[0]):
				for j in xrange(img.shape[1]):
					newimg[i*image_resize_shape[0]:(i+1)*image_resize_shape[0], j*image_resize_shape[1]:(j+1)*image_resize_shape[1]] = img[i][j] * np.ones(image_resize_shape)
			temp.append(newimg)
		result = np.mean(temp, axis=0)
		cv2.imwrite('tmp/'+str(k)+'_'+str(counter)+'_'+ext+'.jpg', result)


window_size = 7
training_epochs = 50
batch_size = 50
file_batch_size = 10
stride = 1

# window_size = 11
# training_epochs = 50
# batch_size = 40
# file_batch_size = 5
# stride = 1
fimgList = getFiles(sys.argv[1])

number_of_file_batch = len(fimgList[0:200]) / file_batch_size
	
W = pickle.load(open('Weights.pickle_W1'))

cl = Conv()

aeL1 = AutoEncoder.AutoEncoder(number_of_inputLayer=window_size*window_size*64, number_of_hiddenLayer=64)
trainL1 = aeL1.fit(corruption_quantity=0.30, learning_rate=0.1, batch_size=batch_size, have_sparsity_penalty=False, type_cost='SQUAREMEAN', output_type_non_linearity='RELU')

for epoch in xrange(training_epochs):

	random.shuffle(fimgList)

	st = time.time()
	c = []
	print 'for epoch ', epoch
	for batch_index_file in xrange(number_of_file_batch):
		images = []
		#print '	reading data in range(', batch_index_file * file_batch_size, ',', (batch_index_file+1) * file_batch_size, ')'
		for fimg in fimgList[batch_index_file * file_batch_size:(batch_index_file+1) * file_batch_size]:
			img = cv2.imread(sys.argv[1]+'/'+fimg)
			img = cv2.resize(img, ((200*img.shape[1])/img.shape[0], 200)) / 255.0
			img = cl.convolve(img, W, feature_shape=(7, 7, 3))
			img = cl.max_pooling(img)
			images.append(img)
		data = create_image_patches(images, patch_size=(window_size, window_size), stride=stride)
		aeL1.add_sample(data)
		if data.shape[0] % batch_size == 0:
			n_train_batches = data.shape[0]/batch_size
		else:
			n_train_batches = (data.shape[0]/batch_size) + 1
		for batch_index in xrange(n_train_batches):
			c.append(trainL1(batch_index))
	fwWeights = open('Weights.pickle_W1_7x7_RELU_'+str(epoch), 'w')
	print 'Training epoch %d, cost ' % epoch, np.mean(c), ' time consumed ', time.time() - st
	test(aeL1.W.get_value(), epoch, 'W1_7x7_RELU_', channel=64)
	W1 = aeL1.W.get_value()
	print >> fwWeights, pickle.dumps(W1)
	fwWeights.close()