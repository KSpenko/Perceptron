# Perceptron

A basic handwritten Python3 implementation of a machine learning perceptron neural network together with a simple keras implementation for cross comparison.

## Description

The project is structured aroud the simplest of neural networks - a Perceptron. The core of the code is a handwritten class that is the perceptron. The neural network is designed according to the the well known mathematical notation using residuals for the learning process. The optimization method used is in the learning process is the SGD. 

At the same time the project also includes a Keras implementation of an (in principal) identical neural network (with the capability to choose different optimization methods). The purpose of this second "keras" perceptron is to compare it and cross validate the functioning of the handwritten code (speed, accuracy).

Both perceptron neural networks include a couple of possible uses, such as: fitting the weights to a training dataset, saving and loading a model with their weights, testing the performance of a model on specific data and plotting incorrect predictions, minimizing an (empty image) input in order to plot a "deep dream" image that minimizes the cost function given a specific ouptput. 

## Getting Started

* Tested on Python 3.6.9.
* Installing the imported modules (keras, tensorflow, numpy, matplotlib)
* Main programs to run are: perceptron.py and keras_perceptron.py

## Help

Code was managed with Visual Studio Code, encountered following issues:
* https://github.com/tensorflow/tensorflow/issues/44467

## Authors

Contributors names and contact info

ex. Krištof Špenko  
ex. [@Kspenko](https://twitter.com/Kspenko)

## License

This project is licensed under the Apache License - see the LICENSE.md file for details