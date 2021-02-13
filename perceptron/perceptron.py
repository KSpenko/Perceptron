from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import random
import os
import matplotlib.pyplot as plt

from functions import sigmoid
from mnist import reformat_data

import save_load

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

#------------------------------------------------------------------------------------------------
""" Perceptron (ASSUMPTION): each neuron in layer L has a connection to every neuron in the previous layer L-1, here L, L-1 are layer indices.
-Perceptron uses for the activation function of neurons the function tanh(x), so that we do not need to calculate its derivatives, 
because of the relation (d/dx)(tanh(x)) = 1 - (tanh(x))^2.
-Perceptron uses for the activation function of neurons the function sigmoid(x), so that we do not need to calculate its derivatives, 
because of the relation (d/dx)(sigmoid(x)) = sigmoid(x)*(1-sigmoid(x))."""

class Perceptron:
	""" Basic Perceptron, which minimizes cost function with SGD. Each epoch is composed out of (nMinibatch) learning steps. 
Each learning step has (minibatchSize) random training pairs of input data x and output data y. """

	def __init__(self, network_layout, function="tanh"):
		""" network_layout is a list of the number of neurons in each layer, 
teh length of the list indicatets the number of layers; first layer is the input(flattened image), last layer is the output(classification 0-9).""" 
		self.weights = []
		self.biases = []		
		self.nLayers = len(network_layout)
		self.layout = network_layout
		self.function = function
		self.func = np.tanh
		if(function=="sigmoid"): self.func = sigmoid
		# weight initializer (normalized random Gaussian)
		for i in range(1, self.nLayers):	
			self.biases.append( np.zeros(self.layout[i]) )
			self.weights.append( np.random.normal(loc=0.0, scale=np.sqrt(2./(self.layout[i]+self.layout[i-1])), size=(self.layout[i], self.layout[i-1]) ) )

	def feedforward(self, input_data, output_data, data_size):
		""" fo the give input we calculate the activations at the nodes in each layer
input_data - one or a list of multiple input vectors,
output_data - one or a list of multiple output vectors for calculating the cost function,
data_size - first dimension of input_data or the number of input vectors,
activations - neurons activations, 
activations[-1] - when the method returns this object, we get output vector/vectors in the similar form as the input vectors in input_data,
estimate - value of the cost function averaged over the whole minibatch."""
		activations = [np.array(input_data)]
		for b, w in zip(self.biases, self.weights):
			#activations.append( self.func( np.dot(w, activations[-1].T) + np.tile(b, (data_size,1)).T ).T )
			activations.append( self.func( np.einsum("kj,ij->ik", w, activations[-1]) + np.tile(b, (data_size,1)) ) )
		estimate = (1./data_size)*np.sum(np.power((output_data - activations[-1]), 2))
		return activations, estimate

	def evaluate(self, input_data, output_data, data_size):
		""" method that, for a given set of data, returns the cost and the accuracy of the model
input_data - one or a list of multiple input vectors,
output_data - one or a list of multiple output vectors for calculating the cost function,
data_size - first dimension of input_data or the number of input vectors,
estimate - value of the cost function averaged over the whole minibatch
accuracy - number of accurate predictions of the model, according to the reference values of the outputs from output_data."""
		activations, estimate = self.feedforward(input_data, output_data, data_size)
		output_prediction = np.argmax(activations[-1], axis=1)
		output_ref = np.argmax(output_data, axis=1)
		accuracy = np.float(np.sum(output_prediction == output_ref)/data_size)
		return estimate, accuracy

	def backpropagation(self, input_data, output_data, data_size, eta=0.001):
		""" for a given set of data we calculate an averaged leaning (correction) step of the weights, 
		returns cost and the accuracy of the model for the given batch before the applying the iterative corrections.
input_data - one or a list of multiple input vectors,
output_data - one or a list of multiple output vectors for calculating the cost function,
data_size - first dimension of input_data or the number of input vectors,
eta - learning parameter (how "fast" does the model learn),
estimate - value of the cost function averaged over the whole minibatch
accuracy - number of accurate predictions of the model, according to the reference values of the outputs from output_data. """
		activations, estimate = self.feedforward(input_data, output_data, data_size)
		if(self.function == "tanh"):
			derivative = 1.-np.power(activations[-1], 2.)	# (d/dx)(tanh(x)) = 1 - (tanh(x))^2
		elif(self.function == "sigmoid"):
			derivative = np.multiply(activations[-1],1-activations[-1])	# (d/dx)(sigmoid(x)) = sigmoid(x)*(1-sigmoid(x))
		residual = [np.multiply(-2.*(output_data-activations[-1]), derivative)]
		for i in range(self.nLayers-2, 0, -1):
			#residual.insert(0, np.multiply(np.dot(np.transpose(self.weights[i]), residual[0].T).T, 1. - np.power(activations[i], 2.)) )
			residual.insert(0, np.multiply( np.einsum("kj,ik->ij", self.weights[i], residual[0]), 1.-np.power(activations[i], 2.)) )
		for i in range(self.nLayers-1):
			self.weights[i] -= (eta/data_size)*np.sum(np.einsum("ik,ij->ikj", residual[i], activations[i]), axis=0)
			self.biases[i] -= (eta/data_size)*np.sum(residual[i], axis=0)
		
		output_prediction = np.argmax(activations[-1], axis=1)
		output_ref = np.argmax(output_data, axis=1)
		accuracy = np.float(np.sum(output_prediction == output_ref)/data_size)
		return estimate, accuracy

	def SGD(self, input, output, val_input, val_output, nEpoch, nMinibatch, minibatchSize, eta=0.001):
		""" SGD - Stohastic Gradient Descent algorithm for the set of data
input - learning set of input vectors,
output - learning set of output vectors for the cost function,
val_input - validation set of input vectors,
val_output - validation set of output vectors,
nEpoch - number of repetative epochs during training,
nMinibatch - number on how many parts we can divide our set of data,
eta - learning parameter (how "fast" does the model learn),
		"""
		if(logging):
			f1.write("#Batch costTraining precTraining\n")
			f2.write("#Batch costTraining precTraining costValidation precValidation\n")

		for i in range(nEpoch):
			indexes = list(range(60000))
			random.shuffle(indexes)
			for j in range(nMinibatch):
				batch_input = [input[index] for index in indexes[j*minibatchSize: (j+1)*minibatchSize] ]
				batch_output = [output[index] for index in indexes[j*minibatchSize: (j+1)*minibatchSize] ]
				batch_cost, batch_acc = self.backpropagation(batch_input, batch_output, minibatchSize, eta)
				if(logging):
					f1.write(str(int(i*nMinibatch + j))+' '+str(batch_cost)+' '+str(batch_acc)+'\n')

			if(logging):
				eval_train = self.evaluate(x_train, y_train, 60000)
				eval_test = self.evaluate(x_test, y_test, 10000)
				f2.write(str((i+1)*nMinibatch)+' ')
				f2.write(' '.join(map(str, eval_train))+' ' )
				f2.write(' '.join(map(str, eval_test))+'\n' )    

	def test(self, input, output, data_size, nm, folder):
		""" Short script for testing model and making an image of incorrect predictons. """
		activations, estimate = self.feedforward(input, output, data_size)
		output_prediction = np.argmax(activations[-1], axis=1)
		output_ref = np.argmax(output, axis=1)
		incorrects = np.argwhere(output_prediction != output_ref)

		sample_list = random.sample(list(incorrects), int(nm[0]*nm[1]))
		fig, ax = plt.subplots(nm[0], nm[1])
		fig.set_size_inches(15, 8)
		for i in range(nm[0]):
			for j in range(nm[1]):
				index = sample_list[i*nm[1]+j]
				ax[i][j].set_title('Predicted: '+str(output_prediction[index])+', Correct: '+str(output_ref[index]))
				ax[i][j].imshow(input[index].reshape(img_rows, img_cols), cmap='Greys')
		fig.suptitle('Cost: '+str(estimate)+', Accuracy: '+str(np.float((data_size-len(incorrects))/data_size)))
		plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.5)
		plt.savefig(folder+'/incorrects.png')
		plt.show()

	def deep_dream(self, number, nEpoch, eta=0.001):
		""" Script to calculate deep dream image of a specific output. 
	Unlike in the Keras implementation we do not add an additional layer but optimize the activations of the input layer. """
		if(act_func=="sigmoid"):
			x_dd = np.array([np.zeros(img_rows*img_cols)])
			y_dd = to_categorical([number], num_classes)
		elif(act_func=="tanh"):
			x_dd = np.array([np.zeros(img_rows*img_cols)-1.])
			y_dd = to_categorical([number], num_classes)*2.-1.
		
		for j in range(nEpoch):
			activations = self.feedforward(x_dd, y_dd, 1)[0]
			if(self.function == "tanh"):
				derivative = 1.-np.power(activations[-1], 2.)	# (d/dx)(tanh(x)) = 1 - (tanh(x))^2
			elif(self.function == "sigmoid"):
				derivative = np.multiply(activations[-1],1-activations[-1])	# (d/dx)(sigmoid(x)) = sigmoid(x)*(1-sigmoid(x))
			residual = np.multiply(-2.*(y_dd-activations[-1]), derivative)
			for i in range(self.nLayers-2, 0, -1):
				residual = np.multiply( np.einsum("kj,ik->ij", self.weights[i], residual), 1.-np.power(activations[i], 2.))
			x_dd[0] -= (eta)*np.sum(np.einsum("ik,kj->ij", residual, self.weights[0]), axis=0)
			if(act_func=="sigmoid"):
				x_dd = np.clip(x_dd, 0., 1.)
			elif(act_func=="tanh"):
				x_dd = np.clip(x_dd, -1., 1.)
		
		score = self.evaluate(x_dd, y_dd, 1)
		return x_dd[0], score

#================================================================================================
if __name__ == '__main__':
	""" Usage:
1. training the model without saving, to estimate were the over-training begins (minimum of cost on validation): switch=True, save=False, logging=True, test=False
2. training the model and saving weights, with an "ideal" number of epochs: switch=True, save=True, logging=False, test=False
3. testing the "ideal" model: switch=False, save=False, logging=False, test=True
4. for testing the deep dream feature: dd = True."""

	nEpoch = 200
	name = 'perceptron'
	folder = 'models/'+name
	
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
		model = Perceptron(layout, function="tanh")
		model.SGD(x_train, y_train, x_test, y_test, nEpoch, nMinibatch, minibatchSize, eta=0.001)
		if(save):
			save_load.save_model(model, file_name=(folder+'/model.xlsx'))
		if(logging):
			f1.close()
			f2.close()

	if(test):
		model = save_load.load_xlsx(file_name=(folder+'/model.xlsx'), layout=layout)
		model.test(x_test, y_test, 10000, (4, 6), folder)

	if(dd):
		model = save_load.load_xlsx(file_name=(folder+'/model.xlsx'), layout=layout)
		fig, ax = plt.subplots(2, 5)
		fig.set_size_inches(15, 6)
		for i in range(num_classes):
			img, score = model.deep_dream(i, 200*nMinibatch, eta=0.001)
			#print(img)
			print(i)
			print(score)
			print(np.amax(img))
			print(np.amin(img))
			im = ax[int(i/5)][int(i%5)].imshow(np.reshape(img, newshape=(img_rows, img_cols)), cmap='Greys')
			ax[int(i/5)][int(i%5)].set_title(str(i))
		plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.5, hspace=0.5)
		plt.savefig(folder+'/deepdream.png')