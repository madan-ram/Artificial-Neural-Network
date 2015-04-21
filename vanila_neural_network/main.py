import NeuralNetwork as nn
import theano
import theano.tensor as T
import numpy as np
from logistic_sgd import LogisticRegression
from numpy import genfromtxt
from sklearn import preprocessing
import time

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.) # acc is allocated for each parameter (p) with 0 values with the shape of p
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def SGD(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append((p, p - lr * g))
    return updates

def dropout(X, p=0., srng = None):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X


params = []

data = genfromtxt('train.csv', delimiter=',')

X = T.matrix('X')
index = T.lscalar('index')
y = T.ivector('y_input')
lr = 0.01

ids, data, target = data[:, 0], data[:, 1:-1], data[:, -1] 

index_shuffle = range(data.shape[0])
np.random.shuffle(index_shuffle)

data = data[index_shuffle]
target = target[index_shuffle]

# data = preprocessing.scale(data)

data_size = data.shape[0]

validation_size = (20*data_size)/100

train_data = data[validation_size:]
train_label = target[validation_size:]


valid_data = data[:validation_size]
valid_label = target[:validation_size]

data = genfromtxt('test.csv', delimiter=',')
ids, data = data[:, 0], data[:, 1:]

test_data = data

batch_size = 1000
train_epoch = 100
c = []

L1 = nn.NNLayer(number_of_input_layer=93, number_of_output_layer=250)
L1_result = L1.__compute__(input=X, non_linearity_fun_type='RELU', enable_dropout=True, p=0.2)
#L1_result = L1.__compute__(input=X, enable_dropout=True, p=0.2)
params += L1.get_params()

L2 = nn.NNLayer(number_of_input_layer=250, number_of_output_layer=250)
L2_result = L2.__compute__(input=L1_result, non_linearity_fun_type='RELU', enable_dropout=True, p=0.5)
#L2_result = L2.__compute__(input=L1_result, enable_dropout=True, p=0.5)
params += L2.get_params()

L3 = nn.NNLayer(number_of_input_layer=250, number_of_output_layer=250)
L3_result = L3.__compute__(input=L2_result, non_linearity_fun_type='RELU', enable_dropout=True, p=0.5)
#L3_result = L3.__compute__(input=L2_result, enable_dropout=True, p=0.5)
params += L3.get_params()


L4 = nn.NNLayer(number_of_input_layer=250, number_of_output_layer=250)
L4_result = L4.__compute__(input=L3_result, non_linearity_fun_type='SIGMOID', enable_dropout=True, p=0.5)
#L3_result = L3.__compute__(input=L2_result, enable_dropout=True, p=0.5)
params += L4.get_params()


fully_L4 = LogisticRegression(input=L4_result, n_in=250, n_out=9)
params += fully_L4.params

cost = fully_L4.negative_log_likelihood(y)
gparams = T.grad(cost, params)



data_shared = theano.shared(np.asarray(train_data, dtype=theano.config.floatX), borrow=True)
train_label = np.asarray(train_label, dtype='int32')
# label_shared = theano.shared(training_label.reshape((training_label.shape[0], 1)), borrow=True)
label_shared = theano.shared(train_label, borrow=True)



data_shared_valid = theano.shared(np.asarray(valid_data, dtype=theano.config.floatX), borrow=True)
label_shared_valid = theano.shared(np.asarray(valid_label, dtype='int32'), borrow=True)

validation = theano.function(
        [],
        fully_L4.errors(y),
        givens = [
            (X, data_shared_valid),
            (y, label_shared_valid)
        ]
    )

data_shared_test = theano.shared(np.asarray(test_data, dtype=theano.config.floatX), borrow=True)

test = theano.function(
        [],
        fully_L4.p_y_given_x,
        givens = [
            (X, data_shared_test)
        ]
    )


number_of_batchs = train_data.shape[0]/batch_size
print 'training started'
for epoch in xrange(train_epoch): 
    if epoch%10 == 0:
        updates = RMSprop(cost, params, lr=lr, rho=0.9, epsilon=1e-6)
        #updates = SGD(cost, params, lr=lr)
        print 'current learning rate', lr
        train = theano.function(
            [index],
            fully_L4.errors(y),
            givens = [
                (X, data_shared[index * batch_size: (index+1) * batch_size]),
                (y, label_shared[index * batch_size: (index+1) * batch_size])
            ],
            updates=updates,
        on_unused_input='warn')
        lr = lr/10.0
    c = []
    for batch_index in xrange(number_of_batchs):
        t = train(batch_index)
        c.append(t)

    print 'epoch', epoch
    print '	training error =>', np.mean(c) * 100, '%'
    print '	validation error =>', validation() * 100, '%'
    print

    # test_f = open('result.txt', 'w')
    # print >> test_f,'id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9'
    # for i,r in enumerate(test()):
    # 	print >> test_f, str(i+1)+','+','.join(map(str, r))
    # 	# temp = [0]*10
    # 	# temp[r] = 1
    # 	# temp[0] = i+1
    # 	# print >> test_f, ','.join(map(str, temp))
    # test_f.close()
    # print 'finished writing result'


    # test_f = open('result.txt', 'w')
    # print >> test_f,'id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9'
    # for i,r in enumerate(test()):
    # 	temp = [0]*10
    # 	temp[r] = 1
    # 	temp[0] = i+1
    # 	print >> test_f, ','.join(map(str, temp))
    # test_f.close()
    # print 'finished writing result'
    # time.sleep(5)
