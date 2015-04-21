import AutoEncoder
import cPickle as pickle
import gzip
import numpy as np
from utils import tile_raster_images
import cv2
import theano
import theano.tensor as T
from logistic_sgd import LogisticRegression
import NeuralNetwork as nn
import warnings

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


def cost_fun(y_hat, y, cost_type='SQUAREMEAN'):
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
#     return (lambada_/2.0)*(T.sum(W**2) + T.sum(W_prime**2) )


#data for training
# f = gzip.open('mnist.pkl.gz', 'rb')
# train_set, valid_set, test_set = pickle.load(f)

# training_data, training_label = train_set

f = open('images.pkl', 'rb')
train_set, valid_set = pickle.load(f)

training_data, training_label = train_set
valid_data, valid_label = valid_set

print 'first layer of autoencoder started ------------------------>'
# #intialize some of the paraameter
window_size = 50
batch_size = 20
train_epoch =  100

#create autoencoder object ,(we are using sparce autoencoder)
aeL1 = AutoEncoder.AutoEncoder(number_of_input_units=window_size*window_size, number_of_hidden_units=500)
trainL1 = aeL1.fit(learning_rate=0.00005, batch_size=batch_size, have_sparsity_penalty=True, cost_type='CROSSENTROPY', learning_type='RMSPROP')

#add the data to be trained
aeL1.add_sample(training_data)

number_of_batchs = training_data.shape[0]/batch_size
print 'training started'
for epoch in xrange(train_epoch):
	c = []
	for batch_index in xrange(number_of_batchs):
		c.append(trainL1(batch_index))
	print 'epoch ->', epoch, 'training reconstruction error ->', np.mean(c)
	test(aeL1.W.get_value(), epoch, 'W1', channel=1)

print 'second layer of autoencoder started ------------------------>'

# #intialize some of the paraameter
# batch_size = 20
# train_epoch = 50

#create autoencoder object ,(we are using sparce autoencoder)
aeL2 = AutoEncoder.AutoEncoder(number_of_input_units=200, number_of_hidden_units=200)
trainL2 = aeL2.fit(learning_rate=0.00005, batch_size=batch_size, have_sparsity_penalty=True, cost_type='CROSSENTROPY', learning_type='RMSPROP')

W1 = aeL1.W.get_value()
data = training_data.dot(W1)

data = (data - np.min(data))/(np.max(data) - np.min(data))

#add the data to be trained
aeL2.add_sample(data)

number_of_batchs = data.shape[0]/batch_size
print 'training started'
for epoch in xrange(train_epoch):
	c = []
	for batch_index in xrange(number_of_batchs):
		c.append(trainL2(batch_index))
	print 'epoch ->', epoch, 'training reconstruction error ->', np.mean(c)

print 'third layer of autoencoder started ------------------------>'

# #intialize some of the paraameter
# batch_size = 20
# train_epoch = 50

#create autoencoder object ,(we are using sparce autoencoder)
aeL3 = AutoEncoder.AutoEncoder(number_of_input_units=200, number_of_hidden_units=200)
trainL3 = aeL3.fit(learning_rate=0.00005, batch_size=batch_size, have_sparsity_penalty=True, cost_type='CROSSENTROPY', learning_type='RMSPROP')

W2 = aeL2.W.get_value()
data = data.dot(W2)

data = (data - np.min(data))/(np.max(data) - np.min(data))

#add the data to be trained
aeL3.add_sample(data)

number_of_batchs = data.shape[0]/batch_size
print 'training started'
for epoch in xrange(train_epoch):
	c = []
	for batch_index in xrange(number_of_batchs):
		c.append(trainL3(batch_index))
	print 'epoch ->', epoch, 'training reconstruction error ->', np.mean(c)

#activation after 3rd layer
W3 = aeL3.W.get_value()











X = T.matrix('X_input')
index = T.lscalar()
y = T.ivector('y_input')
train_epoch = 100
decay_rate = 0.99
step_cache = []
updates = []
params = []
learning_rate = 0.00005
batch_size = 4
lambada_ = 1e-3

L1 = nn.NNLayer(number_of_input_layer=window_size * window_size, number_of_output_layer=500, weights=W1, b=aeL1.b.get_value())
resultL1 = L1.__compute__(input=X)
params += L1.get_params()

L2 = nn.NNLayer(number_of_input_layer=500, number_of_output_layer=200, weights=W2, b=aeL2.b.get_value())
resultL2 = L2.__compute__(input=resultL1)
params += L2.get_params()

L3 = nn.NNLayer(number_of_input_layer=200, number_of_output_layer=200, weights=W3, b=aeL3.b.get_value())
resultL3 = L3.__compute__(input=resultL2)
params += L3.get_params()

# L1 = nn.NNLayer(number_of_input_layer=window_size * window_size, number_of_output_layer=200)
# resultL1 = L1.__compute__(input=X)
# params += L1.get_params()

# L2 = nn.NNLayer(number_of_input_layer=200, number_of_output_layer=200)
# resultL2 = L2.__compute__(input=resultL1)
# params += L2.get_params()

# L3 = nn.NNLayer(number_of_input_layer=200, number_of_output_layer=200)
# resultL3 = L3.__compute__(input=resultL2)
# params += L3.get_params()


# fully_L4 = nn.NNLayer(number_of_input_layer=200, number_of_output_layer=10)
# resultL4 = fully_L4.__compute__(input=resultL3, non_linearity_fun_type='SOFTMAX')
# params += fully_L4.get_params()

fully_L4 = LogisticRegression(input=resultL3, n_in=200, n_out=10)
params += fully_L4.params

cost = fully_L4.negative_log_likelihood(y)

# cost = cost_fun(resultL4, y, cost_type='CROSSENTROPY') #+ regularizer(lambada_)

gparams = T.grad(cost, params)


temp = zip(gparams, params)
for i in xrange(len(temp)):
    gparam, param = temp[i]
    try:
        step_cache[i] = step_cache[i] * decay_rate + (1.0 - decay_rate) * gparam ** 2
        print 'using step_cache'
    except IndexError:
        step_cache.append(T.zeros(gparam.shape) * decay_rate + (1.0 - decay_rate) * gparam ** 2)

    updates.append((param, param - (learning_rate * gparam) / (T.sqrt(step_cache[i] + 1e-8))))


data_shared = theano.shared(np.asarray(training_data, dtype=theano.config.floatX), borrow=True)
training_label = np.asarray(training_label, dtype='int32')
# label_shared = theano.shared(training_label.reshape((training_label.shape[0], 1)), borrow=True)
label_shared = theano.shared(training_label, borrow=True)


train = theano.function(
		[index],
		fully_L4.errors(y),
        givens = [
            (X, data_shared[index * batch_size: (index+1) * batch_size]),
            (y, label_shared[index * batch_size: (index+1) * batch_size])
        ],
        updates=updates
	)


data_shared_valid = theano.shared(valid_data, borrow=True)
label_shared_valid = theano.shared(np.asarray(valid_label, dtype='int32'), borrow=True)

validation = theano.function(
        [],
        fully_L4.errors(y),
        givens = [
            (X, data_shared_valid),
            (y, label_shared_valid)
        ],
        updates=updates
    )

predict_valid = theano.function(
        [],
        fully_L4.y_pred,
        givens = [
            (X, data_shared_valid),
            (y, label_shared_valid)
        ],
        updates=updates
    )

number_of_batchs = training_data.shape[0]/batch_size
print 'training started'
for epoch in xrange(train_epoch):
	c = []
	for batch_index in xrange(number_of_batchs):
		c.append(train(batch_index))
	test(L1.W.get_value(), epoch, 'W1', channel=1)
	print 'epoch ->', epoch, 'validation error ->', np.mean(validation()) * 100, '%' 
	print np.sum(np.asarray(predict_valid()) == np.asarray(valid_label))/float(np.asarray(valid_label).shape[0])
	print 'epoch ->', epoch, 'training error ->', np.mean(c) * 100, '%'
