Joachim-x2-net

Joachim-x2-net is a small PyTorch project that trains a neural network to approximate the mathematical function x². The project demonstrates how to create a simple regression model, normalize data, train the network, save it, and later load it for prediction.

Overview

This project consists of two main scripts:

train.py – trains the neural network on normalized input–output pairs.

predict.py – loads the trained model and predicts the square of a given number.

The model learns to take a single input value and output its squared value using supervised learning.

Project Structure
Joachim-x2-net/
│── train.py        # Trains the neural network
│── predict.py      # Loads the model and predicts x² for any input
│── model.pth       # Saved model after training
│── README.md       # Project documentation

Model Architecture

The neural network is a simple feedforward model with:

1 input neuron

2 hidden layers (48 and 32 neurons)

Tanh activation functions

1 output neuron

Tanh is chosen because x² is a smooth and nonlinear curve. Tanh provides smooth, non-zero gradients across the entire input range, making it suitable for learning curved functions.

Training

The training data consists of:

Input values from 0.01 to 2.00 (normalized by dividing by 100)

Output values equal to those inputs squared (normalized by dividing by 10000)

The training process uses:

MSELoss, because this is a regression task

SGD as the optimizer

A low learning rate (0.0001) for stable training

45,000 epochs for high precision

Run training with:

python3 train.py


After training, the model weights are saved to model.pth.

Prediction

Use predict.py to load the trained model and predict the square of any number:

python3 predict.py


You will be prompted to enter a number, and the model will output its predicted squared value.

Purpose of the Project

This project is intended as a simple but complete demonstration of:

Basic neural network construction

Data normalization

Regression with PyTorch

Saving and loading model weights

Building clean and understandable AI project structure

License

This project is released under the MIT License.
