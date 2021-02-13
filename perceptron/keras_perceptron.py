import os
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops import math_ops
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Layer
from keras.initializers import Zeros, RandomNormal, glorot_normal
from keras.models import load_model
from keras.constraints import Constraint
from keras import backend as K
from keras.callbacks import Callback

from mnist import reformat_data

img_rows, img_cols = 28, 28
num_classes = 10

layout = [784, 250, 10]
act_func = "tanh" #"sigmoid"

nMinibatch = 2000
minibatchSize = 30
nEpoch = 10
eta = 0.01

(x0_train, y0_train), (x0_test, y0_test) = mnist.load_data()
x_train, y_train, x_test, y_test = reformat_data(num_classes, act_func)

#-----------------------------------------------------------------------------------------------
def my_loss_fn(y_true, y_pred):
	"Custom Keras cost function based on squared mean error."
	squared_difference = tf.square(y_true-y_pred)
	return tf.reduce_sum(squared_difference, axis=-1)

class MyCustomCallback(Callback):
	"Logger that logs the validation of model on both training and validation data."
	def __init__(self):
		self.batch_step = 0
		self.epoch_step = 0
		if(logging):
			f1.write("#Batch costTraining precTraining\n")
			f2.write("#Batch costTraining precTraining costValidation precValidation\n")

	def on_train_batch_end(self, batch, logs={}): 
		self.batch_step += 1
		if(logging):
			f1.write(str(self.batch_step)+' '+str(logs["loss"])+' '+str(logs["accuracy"])+'\n')

	def on_epoch_end(self, batch, logs={}): 
		self.epoch_step += nMinibatch
		if(logging):
			eval_train = self.model.evaluate(x_train, y_train, verbose=0)
			eval_test = self.model.evaluate(x_test, y_test, verbose=0)
			f2.write(str(self.epoch_step)+' ')
			f2.write(' '.join(map(str, eval_train))+' ' )
			f2.write(' '.join(map(str, eval_test))+'\n' )    


class Uniform(Constraint):
	"Custom Keras Constraint for weight to be limited in interval [0,1]."
	def __call__(self, w):
		if(act_func=="sigmoid"):
			a, b = 0., 1.
		elif(act_func=="tanh"):
			a, b = -1., 1.
		new_w = w * math_ops.cast(math_ops.greater_equal(w, a), K.floatx())
		return new_w * math_ops.cast(math_ops.less_equal(w, b), K.floatx()) + 1. * math_ops.cast(math_ops.greater_equal(w, b), K.floatx())

class BiasLayer(Layer):
	"Custom Keras Layer composed of only biases, no weights."
	def __init__(self, output_dim, input_dim, **kwargs):
		self.output_dim = output_dim
		self.input_dim = input_dim
		super(BiasLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		self.w = self.add_weight(name='biases', 
		shape=(self.input_dim,), 
		initializer='zeros', 
		trainable=True, 
		constraint=Uniform())
		self.W = K.reshape(self.w, shape=(1, self.input_dim))
		super(BiasLayer, self).build(input_shape)

	def call(self, x):
		return x+self.W

	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.output_dim)

#----------------------------------------------------------------------------------------
def construct_perceptron(opt_method):
	model = Sequential()
	# 1st layer INPUT
	model.add(Dense(layout[1], activation='tanh', input_shape=(img_rows*img_cols,), kernel_initializer=glorot_normal(), bias_initializer='zeros'))
	# Traditional hidden layer
	# model.add(Dense(128, activation='tanh'))
	# Dense layer (classification)
	model.add(Dense(num_classes, activation='tanh',kernel_initializer=glorot_normal(), bias_initializer='zeros'))

	# Compile model
	model.compile(loss=my_loss_fn,
		optimizer=opt_method, 
		metrics=['accuracy'])
	return model
	
def train_model(model, x_train, y_train, x_test, y_test):
	my_callback = MyCustomCallback()
	model.fit(x_train, y_train, 
		epochs=nEpoch, 
		batch_size=minibatchSize,
		verbose=0, 
		# validation_data=(x_test, y_test),
		callbacks=[my_callback])

def test_model(model_name, x_test, y0_test, nm, folder):
	model = load_model(model_name, custom_objects={'my_loss_fn': my_loss_fn})
	incorrects = np.argwhere(model.predict_classes(x_test).reshape((-1,)) != y0_test)
	predictions = model.predict_classes(x_test)
	incorrects = incorrects.reshape((-1,))

	score = model.evaluate(x_test, y_test, verbose=0)

	sample_list = random.sample(list(incorrects), int(nm[0]*nm[1]))
	fig, ax = plt.subplots(nm[0], nm[1])
	fig.set_size_inches(15, 8)
	for i in range(nm[0]):
		for j in range(nm[1]):
			index = sample_list[i*nm[1]+j]
			ax[i][j].set_title('Predicted: '+str(predictions[index])+', Correct: '+str(y0_test[index]))
			ax[i][j].imshow(x0_test[index], cmap='Greys')
	fig.suptitle('Cost: '+str(score[0])+', Accuracy: '+str(score[1]))
	plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.5)
	plt.savefig(folder+'/incorrects.png')
	print(model.count_params())

def deep_dream(model_name, number, nEpoch):
	model = load_model(model_name, custom_objects={'my_loss_fn': my_loss_fn})
	#construct new model with additional input layer
	new_model = Sequential()
	new_model.add(BiasLayer(layout[0], input_dim=img_rows*img_cols))
	#new_model.add(Dense(img_rows*img_cols, input_shape=(img_rows*img_cols,), bias_constraint=Uniform(), name='bias_layer', kernel_initializer='zeros', bias_initializer='zeros'))

	for i in range(len(model.layers)):
		new_model.add(model.get_layer(index=i))	
		new_model.get_layer(index=i+1).trainable=False	
		#new_model.get_layer(index=i+2).trainable=False
		#print(new_model.get_layer(index=i+2).get_weights == model.get_layer(index=i).get_weights)

	new_model.compile(loss=my_loss_fn, 
		optimizer=opt_method, 
		metrics=['accuracy'])

	if(act_func=="sigmoid"):
		x_dd = np.array([np.zeros(img_rows*img_cols)])
		y_dd = to_categorical([number], num_classes)
	elif(act_func=="tanh"):
		x_dd = np.array([np.zeros(img_rows*img_cols)-1.])
		y_dd = to_categorical([number], num_classes)*2.-1.

	new_model.fit(x_dd, y_dd,  
		batch_size=1,
		epochs=nEpoch, 
		verbose=0, 
		validation_data=(x_dd, y_dd))
	score = new_model.evaluate(x_dd, y_dd, verbose=0)
	img = np.copy(new_model.get_layer(index=0).get_weights()[0])
	return img, score

#================================================================================================
if __name__ == '__main__':
	""" Usage:
1. training the model without saving, to estimate were the over-training begins (minimum of cost on validation): switch=True, save=False, logging=True, test=False
2. training the model and saving weights, with an "ideal" number of epochs: switch=True, save=True, logging=False, test=False
3. testing the "ideal" model: switch=False, save=False, logging=False, test=True
4. for testing the deep dream feature: dd = True."""

 	# 200 Epoch is too much, so we can analyze when overfitting occurs
	# 100 for SGD, 50 for ADAM
	nEpoch = 200
	opt_method = 'adam' #'sgd','adam'
	name = 'keras_perceptron'
	folder = 'models/'+name+'_'+opt_method
	model_name=(folder+'/model.h5')
	if not os.path.exists(folder):
		os.makedirs(folder)

	switch = False
	save = False
	global logging
	logging = False
	test = False
	dd = True

	if(switch):
		global f 
		if(logging): 
			f1 = open(folder+'/batch-history.dat', "w")
			f2 = open(folder+'/epoch-history.dat', "w")
		model = construct_perceptron(opt_method)
		train_model(model, x_train, y_train, x_test, y_test)
		if(save):
			model.save(model_name)
		if(logging):
			f1.close()
			f2.close()

	if(test):
		test_model(model_name, x_test, y0_test, (4, 6), folder)

	if(dd):
		fig, ax = plt.subplots(2, 5)
		fig.set_size_inches(15, 6)
		for i in range(num_classes):
			img, score = deep_dream(model_name, i, 200*nMinibatch)
			#print(img)
			print(i)
			print(score)
			print(np.amax(img))
			print(np.amin(img))
			im = ax[int(i/5)][int(i%5)].imshow(np.reshape(img, newshape=(img_rows, img_cols)), cmap='Greys')
			ax[int(i/5)][int(i%5)].set_title(str(i))
		plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.5, hspace=0.5)
		plt.savefig(folder+'/deepdream.png')