import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import theano
import warnings
from theano import function


class AutoEncoder:
    def __init__(self, tied_weights=True, number_of_input_units=100, number_of_hidden_units=50):
    	"""
    		param tied_weights {default => True}: If True then W_prime == W', else W_prime != W' (the weights are untide)
    		param number_of_input_units {default => 100} : number of input units
    		param number_of_hidden_units {default => 50} : number of hidden units
    	"""
        self.tied_weights = tied_weights

        self.X = None
        self.data_shared = None
        self.step_cache = []

        self.number_of_hidden_units = number_of_hidden_units
        self.number_of_input_units = number_of_input_units
        self.number_of_output_layer = self.number_of_input_units

        self.theano_rng = RandomStreams(np.random.randint(2 ** 30))

        r = np.sqrt(6) / np.sqrt(number_of_hidden_units + number_of_input_units + 1)
        
        self.W = theano.shared(
           np.asarray(
                np.random.uniform(low=-r, high=r, size=(number_of_input_units, number_of_hidden_units)),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        self.b = theano.shared(np.zeros(self.number_of_hidden_units, dtype=theano.config.floatX), name='b', borrow=True)
        self.b_prime = theano.shared(np.zeros(self.number_of_output_layer, dtype=theano.config.floatX), name='b_prime',
                                     borrow=True)

        self.params = [self.W, self.b, self.b_prime]

        self.W_prime = None
        if self.tied_weights is False:
            ""
            self.W_prime = theano.shared(
               np.asarray(
                    np.random.uniform(low=-r, high=r, size=(number_of_input_units, number_of_hidden_units)),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
            self.params += [self.W_prime]
        else:
            self.W_prime = self.W.T

    def get_params_for_fully_connected(self):
        return self.params[:-1]

    def get_params(self):
        return self.params

    def KL_divergence(self, x, y):
    	"""
		params x: is Q
		params y: is P
    	"""
        return x * T.log(x / y) + (1 - x) * T.log((1 - x) / (1 - y))

    def sparsity_penalty(self, h, rho=0.1):
    	"""
    	params h: is the activation of hidden layer
    	params rho: is the floating point value where activation as to near to.
    	"""
        rho_hat = T.mean(h, axis=0)
        return T.sum(self.KL_divergence(rho, rho_hat))

    def non_linearity_fun(self, X, type='SIGMOID'):
    	"""
    	params X: is input on which non linearity as to be applied.
    	params type {default => 'SIGMOID'}: specify what type of non linearity function to be used (options => {'SIGMOID', tanh, ReLU}).
    	"""
        if type == 'SIGMOID':
            return T.nnet.sigmoid(X)
        elif type == 'TANH':
            return T.tanh(X)
        elif type == 'RELU':
            return T.maximum(X, 0)
        else:
            warnings.warn("wrong non linearity function !!!, using sigmoid default")
            return T.nnet.sigmoid(X)

    def encode(self, X, type_non_linearity='SIGMOID'):
        return self.non_linearity_fun(T.dot(X, self.W) + self.b, type=type_non_linearity)

    def decode(self, X, linear=False, type_non_linearity='SIGMOID'):
        return self.non_linearity_fun(T.dot(X, self.W_prime) + self.b_prime, type=type_non_linearity)

    def cost_fun(self, X_hat, cost_type='SQUAREMEAN'):
        if cost_type == 'SQUAREMEAN':
            return ((1 / 2.0) * T.mean(T.sum((X_hat - self.X) ** 2, axis=1)))
        elif cost_type == 'CROSSENTROPY':
            return T.mean(-T.sum(self.X * T.log(X_hat) + (1 - self.X) * T.log(1 - X_hat), axis=1))
        else:
            warnings.warn('invalid cost function !!! using corss entropy deafult')
            return T.mean(-T.sum(self.X * T.log(X_hat) + (1 - self.X) * T.log(1 - X_hat), axis=1))

    def rmsprop(self, learning_rate=0.1, decay_rate=0.99, have_sparsity_penalty=False,
                non_linearity_on_output_layer='SIGMOID', non_linearity_on_hidden_layer='SIGMOID', cost_type='SQUAREMEAN',
                corruption_quantity=0.0, lambada_ = 3e-3):
        """
        """
        self.X = T.matrix('X_input', dtype=theano.config.floatX)
        X_noise = self.X * self.theano_rng.binomial(size=self.X.shape, n=1, p=1-corruption_quantity)
        
        h = self.encode(X_noise, type_non_linearity=non_linearity_on_hidden_layer)
        X_hat = self.decode(h, type_non_linearity=non_linearity_on_output_layer)

        cost = self.cost_fun(X_hat, cost_type=cost_type) + self.regularizer(lambada_)

        if have_sparsity_penalty is True:
            cost += self.sparsity_penalty(h) 
        self.gparams = T.grad(cost, self.params)

        updates = []
        temp = zip(self.gparams, self.params)
        for i in xrange(len(temp)):
            gparam, param = temp[i]
            try:
                self.step_cache[i] = self.step_cache[i] * decay_rate + (1.0 - decay_rate) * gparam ** 2
                print 'using step_cache'
            except IndexError:
                self.step_cache.append(T.zeros(gparam.shape) * decay_rate + (1.0 - decay_rate) * gparam ** 2)

            updates.append((param, param - (learning_rate * gparam) / (T.sqrt(self.step_cache[i] + 1e-8))))

        return cost, updates

    def regularizer(self, lambada_):
        """
        """
        return (lambada_/2.0)*(T.sum(self.W**2) + T.sum(self.W_prime**2) )

    def sgd(self, learning_rate=0.1, have_sparsity_penalty=False, 
            non_linearity_on_output_layer='SIGMOID', non_linearity_on_hidden_layer='SIGMOID', 
            cost_type='SQUAREMEAN', corruption_quantity=0.0, lambada_ = 3e-3):
        """
        """
        self.X = T.matrix('X_input', dtype=theano.config.floatX)
        X_noise = self.X * self.theano_rng.binomial(size=self.X.shape, n=1, p=1-corruption_quantity)
        
        h = self.encode(X_noise, type_non_linearity=non_linearity_on_hidden_layer)
        X_hat = self.decode(h, type_non_linearity=non_linearity_on_output_layer)

        cost = self.cost_fun(X_hat, cost_type=cost_type) + self.regularizer(lambada_)

        if have_sparsity_penalty is True:
            cost += self.sparsity_penalty(h) 

        self.gparams = T.grad(cost, self.params)

        updates = []
        for gparam, param in zip(self.gparams, self.params):
            updates.append((param, param - learning_rate * gparam))
        return cost, updates

    def add_sample(self, data):
        self.data_shared.set_value(data)

    def fit(self, learning_rate=0.1, batch_size=100, learning_type='SGD', corruption_quantity=0.0,
            non_linearity_on_output_layer='SIGMOID', non_linearity_on_hidden_layer='SIGMOID', 
            have_sparsity_penalty=False, cost_type='SQUAREMEAN'):

        index = T.lscalar('index')

        if learning_type == 'SGD':
            cost, updates = self.sgd(learning_rate = learning_rate, have_sparsity_penalty=have_sparsity_penalty,
                                     corruption_quantity=corruption_quantity, cost_type=cost_type,
                                     non_linearity_on_output_layer=non_linearity_on_output_layer,
                                     non_linearity_on_hidden_layer=non_linearity_on_hidden_layer)
        elif learning_type == 'RMSPROP':
            cost, updates = self.rmsprop(learning_rate = learning_rate, have_sparsity_penalty=have_sparsity_penalty,
                                         corruption_quantity=corruption_quantity, cost_type=cost_type,
                                         non_linearity_on_output_layer=non_linearity_on_output_layer,
                                         non_linearity_on_hidden_layer=non_linearity_on_hidden_layer)
        else:
            warnings.warn('Invalid learning_type so by default using SGD')
            cost, updates = self.sgd(learning_rate = learning_rate, have_sparsity_penalty=have_sparsity_penalty,
                                     corruption_quantity=corruption_quantity, cost_type=cost_type,
                                     non_linearity_on_output_layer=non_linearity_on_output_layer,
                                     non_linearity_on_hidden_layer=non_linearity_on_hidden_layer)

        self.data_shared = theano.shared(np.zeros((10, 10), dtype=theano.config.floatX), borrow=True)
        train = function(
            [index],
            cost,
            updates=updates,
            givens=[(self.X, self.data_shared[index * batch_size: (index + 1) * batch_size])]
        )
        return train

    def __reset_gpu_memory__(self):
        self.data_shared.set_value([[]])
        self.W.set_value([[]])
        self.b.set_value([])
        self.b_prime.set_value([])
        if self.tied_weights is not True:
            self.W_prime.set_value([[]])