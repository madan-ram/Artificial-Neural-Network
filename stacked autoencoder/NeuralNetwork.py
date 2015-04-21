import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import theano
import warnings
from theano import function


class NNLayer:
    def __init__(self, number_of_input_layer=100, number_of_output_layer=100, weights=None, b=None):
        self.X = T.matrix('X_input', dtype=theano.config.floatX)
        self.y = T.matrix('y_input', dtype=theano.config.floatX)

        self.step_cache = []

        self.number_of_input_layer = number_of_input_layer
        self.number_of_output_layer = number_of_output_layer

        r = np.sqrt(6) / np.sqrt(self.number_of_input_layer + self.number_of_output_layer + 1)
        if weights is None:
            self.W = theano.shared(
               np.asarray(
                    np.random.uniform(low=-r, high=r, size=(self.number_of_input_layer, self.number_of_output_layer)),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
            self.b = theano.shared(np.zeros(self.number_of_output_layer, dtype=theano.config.floatX), name='b', borrow=True)
        else:
            self.W = theano.shared(
               np.asarray(
                    weights,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
            self.b = theano.shared(b, name='b', borrow=True)

        

        self.params = [self.W, self.b]

    def non_linearity_fun(self, X, type='SIGMOID'):
        if type == 'SIGMOID':
            return T.nnet.sigmoid(X)
        elif type == 'TANH':
            return T.tanh(X)
        elif type == 'RELU':
            return T.maximum(X, 0)
        elif type == 'SOFTMAX':
            return T.nnet.softmax(X)
        else:
            warnings.warn("wrong non linearity function !!!, using sigmoid default")
            return T.nnet.sigmoid(X)


    def __compute__(self, input=None, non_linearity_fun_type='SIGMOID'):
        if input is None:
            return self.non_linearity_fun(T.dot(self.X, self.W) + self.b, type=non_linearity_fun_type)
        else:
            self.X = input
            return self.non_linearity_fun(T.dot(self.X, self.W) + self.b, type=non_linearity_fun_type)

    def get_params(self):
        return self.params