import numpy as np

class Initializer(object):
	"""Base class for parameter tensor initializers.
	The :class:`Initializer` class represents a weight initializer used
	to initialize weight parameters in a neural network layer. It should be
	subclassed when implementing new types of weight initializers.
	"""
	def __call__(self, shape):
		"""
		Makes :class:`Initializer` instances callable like a function, invoking
		their :meth:`sample()` method.
		"""
		return self.sample(shape)

	def sample(self, shape):
		"""
		Sample should return a theano.tensor of size shape and data type
		theano.config.floatX.
		Parameters
		-----------
		shape : tuple or int
			Integer or tuple specifying the size of the returned
			matrix.
		returns : theano.tensor
			Matrix of size shape and dtype theano.config.floatX.
		"""
		raise NotImplementedError()

class Uniform(Initializer):
	"""Sample initial weights from the uniform distribution.
	Parameters are sampled from U(a, b).
	Parameters
	----------
	range : float or tuple
		When std is None then range determines a, b. If range is a float the
		weights are sampled from U(-range, range). If range is a tuple the
		weights are sampled from U(range[0], range[1]).
	std : float or None
		If std is a float then the weights are sampled from
		U(mean - np.sqrt(3) * std, mean + np.sqrt(3) * std).
	mean : float
		see std for description.
	"""
	def __init__(self, range=0.01, std=None, mean=0.0):
		if std is not None:
			a = mean - np.sqrt(3) * std
			b = mean + np.sqrt(3) * std
		else:
			try:
				a, b = range  # range is a tuple
			except TypeError:
				a, b = -range, range  # range is a number

		self.range = (a, b)

	def sample(self, shape):
		return np.random.uniform(low=self.range[0], high=self.range[1], size=shape)

class Glorot(Initializer):
	"""Glorot weight initialization.
	This is also known as Xavier initialization [1]_.
	Parameters
	----------
	initializer : lasagne.init.Initializer
		Initializer used to sample the weights, must accept `std` in its
		constructor to sample from a distribution with a given standard
		deviation.
	gain : float or 'relu'
		Scaling factor for the weights. Set this to ``1.0`` for linear and
		sigmoid units, to 'relu' or ``sqrt(2)`` for rectified linear units, and
		to ``sqrt(2/(1+alpha**2))`` for leaky rectified linear units with
		leakiness ``alpha``. Other transfer functions may need different
		factors.
	c01b : bool
		For a :class:`lasagne.layers.cuda_convnet.Conv2DCCLayer` constructed
		with ``dimshuffle=False``, `c01b` must be set to ``True`` to compute
		the correct fan-in and fan-out.
	References
	----------
	.. [1] Xavier Glorot and Yoshua Bengio (2010):
		   Understanding the difficulty of training deep feedforward neural
		   networks. International conference on artificial intelligence and
		   statistics.
	Notes
	-----
	For a :class:`DenseLayer <lasagne.layers.DenseLayer>`, if ``gain='relu'``
	and ``initializer=Uniform``, the weights are initialized as
	.. math::
	   a &= \\sqrt{\\frac{12}{fan_{in}+fan_{out}}}\\\\
	   W &\sim U[-a, a]
	If ``gain=1`` and ``initializer=Normal``, the weights are initialized as
	.. math::
	   \\sigma &= \\sqrt{\\frac{2}{fan_{in}+fan_{out}}}\\\\
	   W &\sim N(0, \\sigma)
	See Also
	--------
	GlorotNormal  : Shortcut with Gaussian initializer.
	GlorotUniform : Shortcut with uniform initializer.
	"""
	def __init__(self, initializer, gain=1.0, c01b=False):
		if gain == 'relu':
			gain = np.sqrt(2)

		self.initializer = initializer
		self.gain = gain
		self.c01b = c01b

	def sample(self, shape):
		if self.c01b:
			if len(shape) != 4:
				raise RuntimeError(
					"If c01b is True, only shapes of length 4 are accepted")

			n1, n2 = shape[0], shape[3]
			receptive_field_size = shape[1] * shape[2]
		else:
			if len(shape) < 2:
				raise RuntimeError(
					"This initializer only works with shapes of length >= 2")

			n1, n2 = shape[:2]
			receptive_field_size = np.prod(shape[2:])

		std = self.gain * np.sqrt(2.0 / ((n1 + n2) * receptive_field_size))
		return self.initializer(std=std).sample(shape)

class GlorotUniform(Glorot):
	"""Glorot with weights sampled from the Uniform distribution.
	See :class:`Glorot` for a description of the parameters.
	"""
	def __init__(self, gain=1.0, c01b=False):
		super(GlorotUniform, self).__init__(Uniform, gain, c01b)


class Constant(Initializer):
	"""Initialize weights with constant value.
	Parameters
	----------
	 val : float
		Constant value for weights.
	"""
	def __init__(self, val=0.0):
		self.val = val

	def sample(self, shape):
		return np.ones(shape) * self.val
		
if __name__ == '__main__':
	print GlorotUniform()((10, 10))