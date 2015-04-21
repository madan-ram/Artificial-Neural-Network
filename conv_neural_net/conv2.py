import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import sys
import os
from os import listdir
from os.path import isfile, join
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SigmoidLayer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from pybrain.datasets import ClassificationDataSet
import math
import multiprocessing as multi
from multiprocessing import Pool
from multiprocessing import Manager

class ConvolutionalLayer():
	verbous = None
	kernels = []
	def __init__(self, verbous=False):
		self.verbous = verbous
		self.kernels = []
		self.maps = []

	def setKernels(self, kernels=[]):
		if kernels != []:
			for x in kernels:
				self.kernels.append(x)
		else:
			self.kernels.append(np.array([[1,1],[-1,-1]]))
			self.kernels.append(np.array([[-1,-1],[1,1]]))
			self.kernels.append(np.array([[1,-1],[1,-1]]))
			self.kernels.append(np.array([[-1,1],[-1,1]]))
			self.kernels.append(np.array([[-1,1],[1,-1]]))
			self.kernels.append(np.array([[1,-1],[-1,1]]))
			self.kernels.append(np.array([[0,1],[1,0]]))
		if self.verbous == True:
			print "kernels for feature mapping"
			for i in xrange(len(self.kernels)):
				print "map", i + 1
				print self.kernels[i]

	def apply(self, inputs):
		self.maps = []
		count = 0
		# temp = []
		for kernel in self.kernels:
			if self.verbous == True:
				print "computation on kernel"
				print kernel
			for input in inputs:
				count += 1
				temp_map = np.zeros((input.shape[0] - kernel.shape[0] + 1, input.shape[1] - kernel.shape[1] + 1))
				for i in xrange(input.shape[0] - kernel.shape[0] + 1):
					for j in xrange(input.shape[1] - kernel.shape[1] + 1):
						temp_map[i, j] = np.sum(input[i:i+kernel.shape[0], j:j+kernel.shape[0]] * kernel)
				temp_map[temp_map<0] = 0
				self.maps.append(temp_map)
				# temp.append(temp_map)
				# if count%3 == 0:
				# 	plt.imshow(np.array(temp[::1]).T)
				# 	# plt.savefig(str(count)+'_1.jpg')
				# 	temp = []
				#temp_map = temp_map.T
				# plt.imshow(temp_map.T, cmap = cm.Greys_r)
				# plt.savefig(str(count)+'_1.jpg')
		return self.maps

	def pooling_max(self, maps, size=(2, 2)):
		self.maps = []
		for map in maps:
			temp_map = np.zeros(((map.shape[0]/ size[0]) if ((map.shape[0]/ float(size[0]))%2 == 0) else ((map.shape[0]/ size[0])+1), (map.shape[1]/ size[1]) if ((map.shape[1]/ float(size[1]))%2 == 0) else ((map.shape[1]/ size[1])+1)))
			for i in xrange(0, map.shape[0], size[0]):
				for j in xrange(0, map.shape[1], size[1]):
					# print i, j, map.shape, temp_map.shape
					temp_map[i/2, j/2] = np.max(map[i:i+size[0],j:j+size[1]])
			self.maps.append(temp_map)
		return self.maps

	def pooling_avg(self, maps, size=(2, 2)):
		self.maps = []
		for map in maps:
			temp_map = np.zeros(((map.shape[0]/ size[0]) if ((map.shape[0]/ float(size[0]))%2 == 0) else ((map.shape[0]/ size[0])+1), (map.shape[1]/ size[1]) if ((map.shape[1]/ float(size[1]))%2 == 0) else ((map.shape[1]/ size[1])+1)))
			for i in xrange(0, map.shape[0], size[0]):
				for j in xrange(0, map.shape[1], size[1]):
					# print i, j, map.shape, temp_map.shape
					temp_map[i/2, j/2] = int(np.mean(map[i:i+size[0],j:j+size[1]]))
			self.maps.append(temp_map)
		return self.maps




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

def runner(path):
	global count
	count += 1
	print "image number", count
	img = cv2.imread(path)
	try:
		img = cv2.resize(img, (50, 65)) / 255
		maps = []
		maps.append(img[:,:,0])
		maps.append(img[:,:,1])
		maps.append(img[:,:,2])

		ConvolutionalLayersList = []
		for x in xrange(2):
			ConvolutionalLayersList.append(ConvolutionalLayer())

		for x in ConvolutionalLayersList:
			x.setKernels()
			maps = x.apply(maps)
			maps = x.pooling_max(maps)
		maps_list.append((maps, path))
		# endCl = ConvolutionalLayer()
		# maps = endCl.pooling_avg(maps)
		#print maps[0].shape
	except Exception as e:
		print e


def testing(net):
	global fileNames_path_valid
	global map_fname_label_valid
	global maps_list
	predicted_y = []
	actual_y = []
	random.shuffle(fileNames_path_valid)
	maps_list = manager.list([])
	pool = multi.Pool(processes=4)
	pool.map(runner, fileNames_path_valid[:100])

	
	for maps, path in maps_list:
		result =  net.activate(np.ravel(np.array(maps)))
		label = map_label_int_label[map_fname_label_valid[path]]
		if result[0] >= 0.5:
			predicted_y.append(1)
		else:
			predicted_y.append(0)
		actual_y.append(label)

	print accuracy_score(actual_y, predicted_y)
	print confusion_matrix(actual_y, predicted_y)


dataset_dir_path = sys.argv[1]
train_dir_path = dataset_dir_path + '/Train'

#--------------------------------#
map_label_int_label = {}
count = 0
labels = []
for label in getImmediateSubdirectories(train_dir_path):
	labels.append(label)
	map_label_int_label[label] = count
	count += 1

map_fname_label = {}
fileNames_path = []
for label in labels:
	path_temp = dataset_dir_path + '/Train/' + label
	for name in getFiles(path_temp):
		fileNames_path.append(path_temp + '/' + name)
		map_fname_label[path_temp + '/' + name] = label



manager = Manager()
net = buildNetwork(147*16*12, int(math.sqrt(147*16*12)), int(math.sqrt(int(math.sqrt(147*16*12)))), 1, outclass=SigmoidLayer)



map_fname_label_valid = {}
fileNames_path_valid = []
for label in labels:
	path_temp = dataset_dir_path + '/Validation/' + label
	for name in getFiles(path_temp):
		fileNames_path_valid.append(path_temp + '/' + name)
		map_fname_label_valid[path_temp + '/' + name] = label



number_of_iteration = 200

maps_list = manager.list([])
for i in xrange(number_of_iteration):
	ds = ClassificationDataSet(147*16*12, 1)
	# print fileNames_path
	random.shuffle(fileNames_path)
	print "iteration number ", i+1
	count = 0
	maps_list = manager.list([])
	pool = multi.Pool(processes=4)
	pool.map(runner, fileNames_path[:200])
	for maps, path in maps_list:
		ds.addSample(np.ravel(np.array(maps)), map_label_int_label[map_fname_label[path]])
	trainer = BackpropTrainer(net, ds)
	print trainer.train()
	testing(net)




# img = cv2.imread('1.jpg')
# #scalling feature
# img = cv2.resize(img, (50, 65))/255

# maps = []
# maps.append(img[:,:,0])
# maps.append(img[:,:,1])
# maps.append(img[:,:,2])

# ConvolutionalLayersList = []
# for x in xrange(2):
# 	ConvolutionalLayersList.append(ConvolutionalLayer())

# for x in ConvolutionalLayersList:
# 	x.setKernels()
# 	maps = x.apply(maps)
# 	maps = x.pooling_max(maps)

# endCl = ConvolutionalLayer()
# maps = endCl.pooling_avg(maps)




# self.maps = cl.apply(self.maps)

# print maps
# print len(maps), maps[0].shape
# count = 0
# for x in maps:
# 	plt.imshow(x, cmap = cm.Greys_r)
# 	plt.savefig('X_'+str(count)+'_1.jpg')
# 	count += 1