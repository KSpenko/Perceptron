from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np

(x0_train, y0_train), (x0_test, y0_test) = mnist.load_data()

def reformat_data(num_classes, name="sigmoid"):
	""" Reformat data: 
x ( [0,255] 2d matrices ---> (sigmoid) [0,1] flattened array / (tanh) [-1,1] flattened array ), 
y ( value ---> (sigmoid) [0,1] array / (tanh) [-1,1] array ) """

	y_train = y0_train.copy()
	y_test = y0_test.copy()
	x_train = []
	x_test = []
	for item in x0_train:
		x_train.append(np.ndarray.flatten(item))
	for item in x0_test:
		x_test.append(np.ndarray.flatten(item))

	if(name=="sigmoid"):
		x_train = (np.array(x_train))/255
		x_test = (np.array(x_test))/255
		y_train = to_categorical(y0_train, num_classes)
		y_test = to_categorical(y0_test, num_classes)
	elif(name=="tanh"):
		x_train = ((np.array(x_train))/255)*2.-1.
		x_test = ((np.array(x_test))/255)*2.-1.
		y_train = to_categorical(y0_train, num_classes)*2.-1.
		y_test = to_categorical(y0_test, num_classes)*2.-1.

	return x_train, y_train, x_test, y_test

def cnn_reformat_data(num_classes, img_rows, img_cols):
	""" Reformat data for keras CNN. """
	x_train = x0_train.copy()
	x_test = x0_test.copy()
	y_train = y0_train.copy()
	y_test = y0_test.copy()

	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

	x_train = x_train/255
	x_test = x_test/255

	y_train = to_categorical(y0_train, num_classes)
	y_test = to_categorical(y0_test, num_classes)

	return x_train, y_train, x_test, y_test