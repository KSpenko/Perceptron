import os
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.initializers import Zeros, RandomNormal
from keras.models import load_model
from keras.callbacks import Callback

from mnist import reformat_data, cnn_reformat_data

img_rows, img_cols = 28, 28
num_classes = 10

layout = [784, 250, 10]
act_func = "tanh" #"sigmoid"

nMinibatch = 500
minibatchSize = 120
nEpoch = 10
eta = 0.01

(x0_train, y0_train), (x0_test, y0_test) = mnist.load_data()

#-----------------------------------------------------------------------------------------------
def my_loss_fn(y_true, y_pred):
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

#----------------------------------------------------------------------------------------
def construct_perceptron(opt_method):
	model = Sequential()
	# 1st layer INPUT
	model.add(Dense(layout[1], activation='tanh', input_shape=(img_rows*img_cols,), kernel_initializer=RandomNormal(stddev=(1./np.sqrt(layout[1]*img_rows*img_cols))), bias_initializer='zeros'))
	# Traditional hidden layer
	# model.add(Dense(128, activation='tanh'))
	# Dense layer (classification)
	model.add(Dense(num_classes, activation='tanh',kernel_initializer=RandomNormal(stddev=(1./np.sqrt(num_classes*layout[1]))), bias_initializer='zeros'))

	# Compile model
	model.compile(loss=my_loss_fn,
		optimizer=opt_method, 
		metrics=['accuracy'])
	return model

def construct_cnn(opt_method):
	model = Sequential()
	# 1st layer INPUT
	model.add(Conv2D(32, kernel_size=(3, 3), 
		activation='relu', 
		input_shape=(img_rows, img_cols, 1)))
	# Convolutional layer
	model.add(Conv2D(64, (3, 3), activation='relu'))
	# Pooling layer
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# Dropout layer (against overfitting)
	model.add(Dropout(0.25))
	# Flattening layer (to get back a 1D array)
	model.add(Flatten())
	# Traditional hidden layer
	model.add(Dense(128, activation='relu'))
	# Dropout layer
	model.add(Dropout(0.5))
	# Dense layer (classification)
	model.add(Dense(num_classes, activation='softmax'))

	# Compile model
	model.compile(loss=my_loss_fn, 
		optimizer=opt_method, 
		metrics=['accuracy'])
	return model
	
def train_model(model, x_train, y_train, x_test, y_test):
	my_callback = MyCustomCallback()
	model.fit(x_train, y_train, 
		batch_size=minibatchSize, 
		epochs=nEpoch, 
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
	plt.show()

#================================================================================================
if __name__ == '__main__':
	""" Usage:
1. training the model without saving, to estimate were the over-training begins (minimum of cost on validation): switch=True, save=False, logging=True, test=False
2. training the model and saving weights, with an "ideal" number of epochs: switch=True, save=True, logging=False, test=False
3. testing the "ideal" model: switch=False, save=False, logging=False, test=True."""

 	#perceptron: 40 optimalno za ADAM, 500 je veliko preveč, dobro da opazimo minimum
	#perceptron: 400 optimalno za SGD, 1000 je veliko preveč, dobro da opazimo minimum
	#cnn: 50 optimalno za ADAM, 100 je veliko preveč, dobro da opazimo minimum
	#cnn: ? optimalno za SGD, 100 je veliko preveč, dobro da opazimo minimum
	nEpoch = 50
	opt_method = 'adam' #'sgd','adam'
	#name = 'keras_perceptron'
	name = 'keras_cnn'
	folder = 'models/'+name+'_'+opt_method
	model_name=(folder+'/model.h5')
	if not os.path.exists(folder):
		os.makedirs(folder)

	#x_train, y_train, x_test, y_test = reformat_data(num_classes, act_func)
	x_train, y_train, x_test, y_test = cnn_reformat_data(num_classes, img_rows, img_cols)

	switch = False
	save = False
	global logging
	logging = False
	test = True

	if(switch):
		global f 
		if(logging): 
			f1 = open(folder+'/batch-history.dat', "w")
			f2 = open(folder+'/epoch-history.dat', "w")
		#model = construct_perceptron(opt_method)
		model = construct_cnn(opt_method)
		train_model(model, x_train, y_train, x_test, y_test)
		if(save):
			model.save(model_name)
		if(logging):
			f1.close()
			f2.close()

	if(test):
		test_model(model_name, x_test, y0_test, (4, 6), folder)