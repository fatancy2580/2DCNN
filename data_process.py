from tensorflow.examples.tutorials.mnist import input_data as mnist_data
import cPickle

from tensorflow.contrib.keras.python.keras.datasets.cifar import load_batch
from tensorflow.contrib.keras.python.keras.utils.data_utils import get_file
import tensorflow as tf
import pdb
import os
import numpy as np
def label_trans(label):
	assert not np.any(np.sum(label,0)==0) , 'not enough batch_size'
	s_m = np.mat(np.dot(label.T,label))
	return np.dot(label,np.power(np.array(s_m.I),0.5))
class cifar10_data_class:
	def __init__(self):
		self.cur_train_num = 0
		self.cur_test_num = 0
		self.num_train_samples = 50000
		self.num_test_samples = 10000
		self.x_train,self.y_train,self.x_test,self.y_test = self.load_data()
		self.class_num = 10
		self.img_size = (32,32)

	def load_data(self):
		"""Loads CIFAR10 dataset.
		Returns:
		  Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
		"""
		# dirname = 'cifar-10-batches-py'
		# origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
		# path = get_file(dirname, origin=origin, untar=True)
		path = '/home/opt603a/qzq/data/cifar-10-batches-py'


		x_train = np.zeros((self.num_train_samples, 3, 32, 32), dtype='uint8')
		y_train = np.zeros((self.num_train_samples,), dtype='uint8')

		for i in range(1, 6):
			fpath = os.path.join(path, 'data_batch_' + str(i))
			data, labels = load_batch(fpath)
			x_train[(i - 1) * 10000:i * 10000, :, :, :] = data
			y_train[(i - 1) * 10000:i * 10000] = labels

		fpath = os.path.join(path, 'test_batch')
		x_test, y_test = load_batch(fpath)

		y_train = np.reshape(y_train, (len(y_train), 1))
		y_test = np.reshape(y_test, (len(y_test), 1))


		x_train = x_train.transpose(0, 2, 3, 1)
		x_test = x_test.transpose(0, 2, 3, 1)

		return x_train, y_train, x_test, y_test
	def next_train_batch(self,batch_size,one_hot = True):
		if self.cur_train_num+batch_size > self.num_train_samples:
			self.cur_train_num = 0
		cur = self.cur_train_num
		self.cur_train_num += batch_size
		target = self.y_train[cur:cur+batch_size]
		if one_hot:
			one_hot_target = np.eye(10)[target]
			return self.x_train[cur:cur+batch_size,:,:,:], one_hot_target.reshape(batch_size,10)
		else:
			return self.x_train[cur:cur+batch_size,:,:,:],target.reshape[batch_size,1]
	def next_test_batch(self,batch_size,one_hot = True):	
		if self.cur_test_num+batch_size > self.num_test_samples:
			self.cur_test_num = 0
		cur = self.cur_test_num
		self.cur_test_num += batch_size
		target = self.y_test[cur:cur+batch_size]
		if one_hot:
			one_hot_target = np.eye(10)[target]
			return self.x_test[cur:cur+batch_size,:,:,:],one_hot_target.reshape(batch_size,10)
		else:
			return self.x_test[cur:cur+batch_size,:,:,:],target.reshape[batch_size,1]

mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
cifar = cifar10_data_class()
if __name__ == '__main__':
	cifardata = cifar10_data_class()
	aaa,bbb = cifardata.next_test_batch(100)

	ccc,ddd = cifardata.next_train_batch(10)
	pdb.set_trace()
	print label_trans(d)
