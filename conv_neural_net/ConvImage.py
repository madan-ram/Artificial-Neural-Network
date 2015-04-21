import sys
import os
from os import listdir
import numpy as np
from os.path import isfile, join
import cv2
import pickle
import random
import scipy as sp
from scipy import ndimage
import time
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.lib import stride_tricks

class Conv:
	def convolve(self, img, W, feature_shape, avg_output = False, origin=0):
		result = []
		for w in W.T:
			w = w.reshape(feature_shape)
			newimg = ndimage.convolve(img, w, mode='constant', cval=0.0, origin=origin)
			result.append(newimg.T)
		if avg_output:
			return np.asarray(np.mean(result, axis=1)).T
		return np.asarray(np.sum(result, axis=1)).T

	def convolve_batch(self, imgs, W, feature_shape, avg_output = False, origin=0):
		result = []
		for img in imgs:
			temp = []
			for w in W.T:
				w = w.reshape(feature_shape)
				newimg = ndimage.convolve(img, w, mode='constant', cval=0.0, origin=origin)
				temp.append(newimg.T)
			result.append(np.mean(temp, axis=1).T)
		return np.asarray(result)

	def max_pooling(self, old_map, size=(2, 2), pad_value = 0):
		old_map = old_map.T
		_, row, col = old_map.shape
		pad_row = size[0] - (row % size[0])
		pad_col = size[1] - (col % size[1])
		if pad_row == size[0]:
			pad_row = 0
		if pad_col == size[0]:
			pad_col = 0
		new_map = []
		for t in  np.ndindex((old_map.shape[0])):
			t = t[0]
			X = old_map[t]
			pad_row_data = np.empty((pad_row, X.shape[1]))
			pad_row_data.fill(pad_value)
			X = np.concatenate((X, pad_row_data), axis=0)

			pad_col_data = np.empty((X.shape[0], pad_col))
			pad_col_data.fill(pad_value)
			X = np.concatenate((X, pad_col_data), axis=1)
			segment_X = stride_tricks.as_strided(X, shape=(X.shape[0]/size[0], X.shape[1]/size[1], size[0], size[1]), strides=(X.strides[0] * size[0], X.strides[1] * size[1]) + X.strides)
			segment_X = segment_X.reshape(X.shape[0]/size[0], X.shape[1]/size[1], size[0] * size[1])
			new_map.append(np.max(segment_X, axis=2).T)
		return np.asarray(new_map).T


def getFiles(dir_path):
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

# filePath_list = []
# for f in getFiles(sys.argv[1]):
# 	filePath_list.append(f)

# random.shuffle(filePath_list)

# no_image = 100
# W1 = pickle.load(open('Weights.pickle_W1_5'))
# cl = Conv()
# imgs = []
# for f in filePath_list[0:no_image]:
# 	img = cv2.imread(sys.argv[1]+'/'+f) / 255.0
# 	imgs.append(img)

# print "--------------------------------------------------------------------------------"
# maps = np.asarray(imgs)
# imgs = []
# for i in xrange(len(maps)):
# 	st = time.time()
# 	l = cl.convolve(maps[i], W1, feature_shape=(7, 7, 3))
# 	l = cl.max_pooling(l)
# 	imgs.append(l)
# 	print time.time() - st
# maps = np.asarray(imgs)
# print "--------------------------------------------------------------------------------"

# print maps.shape

# # imgs = []
# # for i in xrange(len(maps)):
# #	 imgs.append(cl.max_pooling(maps[i]))

# # maps = np.asarray(imgs)
# # print maps.shape
# # for i in xrange(len(maps)):
# #	 counter = 0
# #	 for img in maps[i].T:
# #		 #cv2.imwrite('result/'+str(counter)+'_'+f, img.T * 255)
# #		 plt.imshow(img, cmap = cm.Greys_r)
# #		 plt.savefig('result/'+str(counter)+'_'+f)
# #		 counter += 1