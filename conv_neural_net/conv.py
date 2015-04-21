import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression
# from mlp import HiddenLayer
import cv2
import cPickle, gzip
from utils import tile_raster_images

def test(Weights, counter, ext, channel=1):
    """this is an utility that takes weights and plot there feature as image"""
    tile_shape = (10, 10)
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

class ConvolutionalNeuralNet:
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), weights = None):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        # pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))

        # initialize weights with random weights or if weights != None use this as pretrained weights
        if weights is None:
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                np.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        else:
        	self.W = theano.shared(
        		 np.asarray(
        		 	weights,
        		 	dtype=theano.config.floatX
        		 ),
        		 borrow=True
        	)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        # pooled_out = downsample.max_pool_2d(
        #     input=conv_out,
        #     ds=poolsize,
        #     ignore_border=True
        # )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        #self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


def cost_fun(self, y, y_hat, cost_type='SQUAREMEAN'):
    if cost_type == 'SQUAREMEAN':
        return ((1 / 2.0) * T.mean(T.sum((y_hat - y) ** 2, axis=1)))
    elif cost_type == 'CROSSENTROPY':
        return T.mean(-T.sum(y * T.log(y_hat) + (1 - y) * T.log(1 - y_hat), axis=1))
    else:
        warnings.warn('invalid cost function !!! using corss entropy deafult')
        return T.mean(-T.sum(y * T.log(y_hat) + (1 - y) * T.log(1 - y_hat), axis=1))


# def regularizer(self, lambada_):
#     """
#     """
#     return (lambada_/2.0)*(T.sum(W**2))



f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
train_set, label = train_set[0], train_set[1]
train_set = train_set.reshape((train_set.shape[0], 1, 28, 28))

valid_set, valid_label = valid_set[0], valid_set[1]
valid_set = valid_set.reshape((valid_set.shape[0], 1, 28, 28))

batch_size = 20
training_epoch = 50
window_size = 28
X = T.tensor4('X')
index = T.lscalar()
y = T.ivector('y')

data_shared = theano.shared(np.asarray(train_set, dtype=theano.config.floatX), borrow=True)
label_shared = theano.shared(np.asarray(label, dtype='int32'), borrow=True)

data_shared_valid = theano.shared(np.asarray(valid_set, dtype=theano.config.floatX), borrow=True)
label_shared_valid = theano.shared(np.asarray(valid_label, dtype='int32'), borrow=True)

weights = cPickle.load(open('mnist.pickle_W1_49')).reshape((28, 28, 1, 300))
weights = np.transpose(weights, (3, 2, 0, 1))

params = []

conv_layer_1 = ConvolutionalNeuralNet(np.random, X, (300, 1, window_size, window_size), (train_set.shape[0], 1, 28, 28), weights=weights)
params += conv_layer_1.params

output = conv_layer_1.output.reshape((conv_layer_1.output.shape[0], 300))

# classify the values of the fully-connected sigmoidal layer
fully_connected_layer_1 = LogisticRegression(input=output, n_in=300, n_out=64)
params += fully_connected_layer_1.params

print fully_connected_layer_1.output.shape

# classify the values of the fully-connected sigmoidal layer
fully_connected_layer_2 = LogisticRegression(input=fully_connected_layer_1.output, n_in=64, n_out=10)
params += fully_connected_layer_2.params

cost = fully_connected_layer_2.negative_log_likelihood(y)

gparams = T.grad(cost, params)

learning_rate = 0.1

updates = []
for gparam, param in zip(gparams, params):
    updates.append((param, param - learning_rate * gparam))

train = theano.function(
        [index],
        fully_connected_layer_2.errors(y),
        givens = [
            (X, data_shared[index * batch_size: (index+1) * batch_size]),
            (y, label_shared[index * batch_size: (index+1) * batch_size])
        ],
        updates=updates
	)


validation = theano.function(
        [],
        fully_connected_layer_2.errors(y),
        givens = [
            (X, data_shared_valid),
            (y, label_shared_valid)
        ],
        updates=updates
    )

number_of_batch_iteration = train_set.shape[0] / batch_size


for epoch in xrange(training_epoch):
    c = []
    print 'epoch', epoch
    for batch_index in xrange(number_of_batch_iteration):
        #print 'batch_index', batch_index
        t = train(batch_index)
        c.append(t)
    print 'error for epoch ', epoch, ' is', np.mean(c) * 100.0 ,'%'
    print 'error on validation for epoch', epoch,' is', validation() * 100.0 ,'%'
    W = np.transpose(conv_layer_1.W.get_value(), (2, 3, 1, 0))
    W = W.reshape(W.shape[0] * W.shape[1], W.shape[3])
    test(W, epoch, '_conv_W1_mnist_', channel=1)


# cost = self.cost_fun(y, y_hat, cost_type=cost_type) #+ self.regularizer(lambada_)