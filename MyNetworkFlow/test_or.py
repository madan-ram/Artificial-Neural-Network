import os, sys
import numpy as np
from layers import Dense, Variable, L2_norm
import execute

if __name__ == '__main__':
	data = Variable((None, 2), 'input')

	lable = Variable((None, 1), 'lable')
	l1 = Dense(data, 2, 'full1', activation='sigmoid')
	l2 = Dense(l1, 1, 'full2', activation='tanh')
	l2_loss = L2_norm(lable, l2, 'loss')

	label_val = np.asarray([[-1], [1], [1], [-1]])
	data_val = np.asarray([
		[0, 0],
		[0, 1],
		[1, 0],
		[1, 1]
	])

	for i in xrange(100):
		execute.run([l2_loss], feed_dict={data: data_val, lable: label_val})
		l2_loss.backward()
		l2_loss.update()
	
	print execute.run([l2], feed_dict={data: data_val})



	lable = Variable((None, 1), 'lable')
	l1 = Dense(data, 2, 'full1', activation='sigmoid')
	l2 = Dense(l1, 1, 'full2', activation='sigmoid')
	l2_loss = L2_norm(lable, l2, 'loss')
	label_val = np.asarray([[0], [1], [1], [0]])

	data_val = np.asarray([
		[0, 0],
		[0, 1],
		[1, 0],
		[1, 1]
	])

	for i in xrange(100):
		execute.run([l2_loss], feed_dict={data: data_val, lable: label_val})
		l2_loss.backward()
		l2_loss.update()

	print execute.run([l2], feed_dict={data: data_val})


	lable = Variable((None, 1), 'lable')
	l1 = Dense(data, 2, 'full1', activation='linear')
	l2 = Dense(l1, 1, 'full2', activation='linear')
	l2_loss = L2_norm(lable, l2, 'loss')
	label_val = np.asarray([[0], [1], [1], [0]])

	data_val = np.asarray([
		[0, 0],
		[0, 1],
		[1, 0],
		[1, 1]
	])

	for i in xrange(100):
		execute.run([l2_loss], feed_dict={data: data_val, lable: label_val})
		l2_loss.backward()
		l2_loss.update()

	print execute.run([l2], feed_dict={data: data_val})
