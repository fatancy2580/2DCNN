import tensorflow as tf
import numpy as np
import data_process
import pdb
# define the class of tfLDA
class tfLDA:
	def __init__(self,data_size,batch_size,class_num,inchanel):
		# two different settings as the paper
		self.simple_weights = {
			'conv1w': tf.get_variable('conv1w',[3,3,inchanel,32],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
			'conv2w': tf.get_variable('conv2w',[3,3,32,64],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
			'conv1b': tf.get_variable('conv1b',[32],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
			'conv2b': tf.get_variable('conv2b',[64],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
			'fc1w'	:tf.get_variable('fc1w',[(data_size[0]//4)*(data_size[1]//4)*64,64],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
			'fc1b'	: tf.get_variable('fc1b',[64],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
			'optw'	: tf.get_variable('optw',[64,class_num],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
			'optb'	: tf.get_variable('optb',[class_num],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
		}
		self.complex_weights = {
			'conv1_1w' : tf.get_variable('conv1_1w',[3,3,inchanel,64],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
			'conv1_2w' : tf.get_variable('conv1_2w',[3,3,64,64],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
			'conv2_1w' : tf.get_variable('conv2_1w',[3,3,64,96],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
			'conv2_2w' : tf.get_variable('conv2_2w',[3,3,96,96],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
			'conv3_1w' : tf.get_variable('conv3_1w',[3,3,96,256],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
			'conv3_2w' : tf.get_variable('conv3_2w',[1,1,256,256],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
			'conv3_3w' : tf.get_variable('conv3_3w',[1,1,256,64],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
			'conv1_1b' : tf.get_variable('conv1_1b',[64],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
			'conv1_2b' : tf.get_variable('conv1_2b',[64],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
			'conv2_1b' : tf.get_variable('conv2_1b',[96],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
			'conv2_2b' : tf.get_variable('conv2_2b',[96],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
			'conv3_1b' : tf.get_variable('conv3_1b',[256],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
			'conv3_2b' : tf.get_variable('conv3_2b',[256],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
			'conv3_3b' : tf.get_variable('conv3_3b',[64],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
			'fc1wc'   : tf.get_variable('fc1wc',[(data_size[0]//4)*(data_size[1]//4)*64,64],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
			'fc1bc'   : tf.get_variable('fc1bc',[64],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
			'optwc'   : tf.get_variable('optwc',[64,class_num],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
			'optbc'   : tf.get_variable('optbc',[class_num],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
		}
	# two different loss as the paper
	def sorfmax_loss(self,predicts,labels,class_num):  
		#predicts=tf.nn.softmax(predicts)  
		#labels=tf.one_hot(tf.cast(labels,tf.int32),class_num)
		#loss =-tf.reduce_mean(labels * tf.log(predicts))
		loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=predicts)  
		self.cost = tf.reduce_mean(loss)

		return self.cost
	def F_loss(self,predicts,labels,gamma,sim_net):
		if sim_net:
			loss = self.F_norm_square(predicts - labels) + gamma * self.F_norm_square(self.simple_weights['optw'])
		else:
			loss = self.F_norm_square(predicts - labels) + gamma * self.F_norm_square(self.complex_weights['optwc'])
		return loss
	def F_norm_square(self,t):
		return tf.reduce_mean(tf.square(t))
	# inference two kinds of networks
	def inference_simple(self,images):
		conv1 = tf.nn.bias_add(tf.nn.conv2d(images, self.simple_weights['conv1w'], strides=[1, 1, 1, 1],padding = 'SAME'),
		    self.simple_weights['conv1b'])
		relu1= tf.nn.relu(conv1)
		pool1=tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		conv2 = tf.nn.bias_add(tf.nn.conv2d(pool1,self.simple_weights['conv2w'],strides=[1, 1, 1, 1],padding = 'SAME'),
		    self.simple_weights['conv2b'])
		relu2= tf.nn.relu(conv2)  
		pool2=tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 

		ft = tf.reshape(pool2, [-1, self.simple_weights['fc1w'].get_shape().as_list()[0]])
		fc1 = tf.matmul(ft,self.simple_weights['fc1w'])+self.simple_weights['fc1b']
		fc_relu1=tf.nn.relu(fc1)
		opt = tf.matmul(fc_relu1,self.simple_weights['optw'])+self.simple_weights['optb']
		return (fc1,opt)
	def inference_complex(self,images,phase,itera,keep_prob,bn = True):

		conv1_1 = tf.nn.conv2d(images, self.complex_weights['conv1_1w'], strides=[1, 1, 1, 1],padding = 'SAME')
		if bn:
			conv1_1bn,update_ema1 = self.batchnorm(conv1_1,phase,itera,self.complex_weights['conv1_1b'])
		else:
			update_ema1 = tf.constant([1])
			conv1_1bn = conv1_1

		relu1_1 = tf.nn.relu(conv1_1bn)

		conv1_2 = tf.nn.conv2d(relu1_1, self.complex_weights['conv1_2w'], strides=[1, 1, 1, 1],padding = 'SAME')
		if bn:
			conv1_2bn,update_ema2 = self.batchnorm(conv1_2,phase,itera,self.complex_weights['conv1_2b'])
		else:
			update_ema2 = tf.constant([1])
			conv1_2bn = conv1_2
		relu1_2 = tf.nn.relu(conv1_2bn)

		pool1 = tf.nn.max_pool(relu1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

		conv2_1 = tf.nn.conv2d(pool1, self.complex_weights['conv2_1w'], strides=[1, 1, 1, 1],padding = 'SAME')
		if bn:
			conv2_1bn,update_ema3 = self.batchnorm(conv2_1,phase,itera,self.complex_weights['conv2_1b'])
		else:
			conv2_1bn = conv2_1
			update_ema3 = tf.constant([1])
		relu2_1 = tf.nn.relu(conv2_1bn)

		conv2_2 = tf.nn.conv2d(relu2_1, self.complex_weights['conv2_2w'], strides=[1, 1, 1, 1],padding = 'SAME')
		if bn:
			conv2_2bn,update_ema4 = self.batchnorm(conv2_2,phase,itera,self.complex_weights['conv2_2b'])
		else:
			conv2_2bn = conv2_2
			update_ema4  =tf.constant([1])
		relu2_2 = tf.nn.relu(conv2_2bn)

		pool2 = tf.nn.max_pool(relu2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

		conv3_1 = tf.nn.conv2d(pool2, self.complex_weights['conv3_1w'], strides=[1, 1, 1, 1],padding = 'SAME')
		if bn:
			conv3_1bn,update_ema5 = self.batchnorm(conv3_1,phase,itera,self.complex_weights['conv3_1b'])
		else:
			conv3_1bn = conv3_1
			update_ema5 = tf.constant([1])
		relu3_1 = tf.nn.relu(conv3_1bn)

		conv3_2 = tf.nn.conv2d(relu3_1, self.complex_weights['conv3_2w'], strides=[1, 1, 1, 1],padding = 'SAME')
		if bn:
			conv3_2bn,update_ema6 = self.batchnorm(conv3_2,phase,itera,self.complex_weights['conv3_2b'])
		else:
			conv3_2bn = conv3_2
			update_ema6 = tf.constant([1])
		relu3_2 = tf.nn.relu(conv3_2bn)

		conv3_3 = tf.nn.conv2d(relu3_2, self.complex_weights['conv3_3w'], strides=[1, 1, 1, 1],padding = 'SAME')
		if bn:
			conv3_3bn,update_ema7 = self.batchnorm(conv3_3,phase,itera,self.complex_weights['conv3_3b'])
		else:
			conv3_3bn = conv3_3
			update_ema7 = tf.constant([1])
		relu3_3 = tf.nn.relu(conv3_3bn)

		ft = tf.reshape(relu3_3, [-1, self.complex_weights['fc1wc'].get_shape().as_list()[0]])
		fc1 = tf.matmul(ft,self.complex_weights['fc1wc'])+self.complex_weights['fc1bc']
		fc1dp = tf.nn.dropout(fc1,keep_prob)
		fc_relu1=tf.nn.relu(fc1dp)
		opt = tf.matmul(fc_relu1,self.complex_weights['optwc'])+self.complex_weights['optbc']
		update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4,update_ema5,update_ema6,update_ema7)
		return (fc1,opt,update_ema)
	def batchnorm(self,Ylogits, is_test, iteration, offset, convolutional=True):
		exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
		bnepsilon = 1e-5
		if convolutional:
			mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
		else:
			mean, variance = tf.nn.moments(Ylogits, [0])
		update_moving_everages = exp_moving_avg.apply([mean, variance])
		m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
		v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
		Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
		return Ybn, update_moving_everages
	def optimizer(self,loss,lr=0.0001):
		#train_optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)
		train_optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
		return train_optimizer

if __name__ == '__main__':

	dataset='mnist'	
	#settings for different datasets
	if dataset=='mnist':
		disp_step = 5
		test_step = 20
		train_num = 60000
		test_num = 10000
		batch_size = 200
		learning_rate = 0.001
		sim_net = True
		batch_num = test_num//batch_size
		net = tfLDA((28,28),batch_size,10,1)
		images =  tf.placeholder(tf.float32, [batch_size, 28, 28, 1])
	elif dataset == 'cifar':
		disp_step = 5	
		test_step = 20
		train_num = 50000
		test_num = 10000
		batch_size = 200
		learning_rate = 0.00001
		sim_net = False
		batch_num = test_num//batch_size
		net = tfLDA((32,32),batch_size,10,3)
		images =  tf.placeholder(tf.float32, [batch_size, 32, 32, 3])

	# input 
	labels =  tf.placeholder(tf.float32, [batch_size,10])
	phase  =  tf.placeholder(tf.bool)
	itera  =  tf.placeholder(tf.int32)
	keep_prob = tf.placeholder(tf.float32)


	# inference
	best_test_acc = 0
	if sim_net:
		fc1,opt = net.inference_simple(images)
	else:
		fc2,opt,update_ema = net.inference_complex(images,phase,itera,keep_prob)

	loss = net.sorfmax_loss(opt,labels,10)
	#loss = net.F_loss(opt,labels,1e-4,sim_net)
	# optim
	optimizer = net.optimizer(loss,lr = learning_rate)

	correct_prediction = tf.equal(tf.argmax(opt, 1), tf.argmax(labels, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	init = tf.global_variables_initializer()

	# record the data in tensorboard
	tf.summary.scalar('loss', loss)
	tf.summary.scalar('accuracy', accuracy)
	tf.summary.histogram('optc', net.complex_weights['optwc'])
	tf.summary.histogram('opt', net.simple_weights['optw'])
	f = open('./tr_mnist_softmax.txt','w')

	merged = tf.summary.merge_all()
	g_cnt = 0
	#lunch the session and run
	with tf.Session() as sess:
		sess.run(init)
		train_writer = tf.summary.FileWriter('./summary' + '/tr_mnist_f_loss_nor1', sess.graph)
		for i in range(5000):

			if dataset=='mnist':
				im,la=data_process.mnist.train.next_batch(batch_size)
				#la = data_process.label_trans(la)
			elif dataset == 'cifar':
				im,la = data_process.cifar.next_train_batch(batch_size) 


			if sim_net:
				[_,l,acc,ss] = sess.run([optimizer,loss,accuracy,merged],{images : im,labels: la})
			else:
				[_,l,acc,_,ss] = sess.run([optimizer,loss,accuracy,update_ema,merged],{images : im,labels: la, phase : False, keep_prob : 0.75,itera : i})
				# sess.run(update_ema, {images : im,labels: la, phase : False, keep_prob : 0.75,itera : i})
			f.write(str(l)+'\n')
			if i>10:
				break
			train_writer.add_summary(ss,g_cnt)
			g_cnt += 1
			if i% disp_step ==0 :
				print('step: %d    train loss: %f   train acc: %f' % (i,l,acc))

			if i% test_step == 0:
				total_acc = 0
				for j in range(batch_num):
					# get data
					if dataset=='mnist':
						im,la=data_process.mnist.test.next_batch(batch_size)
					elif dataset=='cifar':
						im,la = data_process.cifar.next_test_batch(batch_size)

					# test
					if sim_net:
						[l,acc,ss] = sess.run([loss,accuracy,merged],{images : im,labels: la})
					else:
						[l,acc,ss] = sess.run([loss,accuracy,merged],{images : im,labels: la, phase : True, keep_prob : 1.0,itera : i})
					total_acc += acc

				best_test_acc = total_acc if total_acc > best_test_acc else best_test_acc
				print('test loss: %f   test acc: %f' % (l,total_acc/batch_num))
				print('best test acc: %f' % (best_test_acc/batch_num))
		train_writer.close()
		f.close()