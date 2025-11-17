Joachim-x2-net

Joachim-x2-net is a small PyTorch project that trains a neural network to approximate the mathematical function x².
The project demonstrates dataset creation, normalization, model design, training, saving, and prediction.

Overview

This repository contains two scripts:

train.py – trains the neural network on normalized input/output values.

predict.py – loads the trained model and predicts the square of a given number.

The model learns to take a single floating-point input and output the approximated squared value.

Project Structure
<pre>
Joachim-x2-net/
│── train.py        # Trains the neural network
│── predict.py      # Loads the model and predicts x²
│── model.pth       # Saved model weights (created after training)
│── README.md       # Documentation
</pre>
Model Architecture

The neural network is a simple feedforward model:
<pre>
Input layer: 1 neuron

Hidden layer 1: 48 neurons, Tanh

Hidden layer 2: 32 neurons, Tanh

Output layer: 1 neuron
</pre>
Tanh is used because x² is a smooth nonlinear function. Tanh provides smooth, non-zero gradients across the input range, making the curve easier for the network to approximate.

Training Details

The network is trained on:

Inputs from 0.01 to 2.00 (generated as i/100)

Targets computed as (i²)/10000

Loss function: MSELoss

Optimizer: SGD

Learning rate: 0.0001

Epochs: 45,000

Training will automatically save the model as model.pth after it finishes.

How to Run the Project
1. Install PyTorch (if not installed)
pip install torch

2. Train the model

Run the training script:

<pre> python3 train.py </pre>


This will:

Create the dataset

Train the neural network

Save the trained weights to model.pth

After training, you will see output like:

Epoch: 44000 | Loss: 0.000002

3. Run predictions

Once model.pth exists, you can run:

python3 predict.py


The script will ask you to enter a number, for example:

Enter a number to square: 2
The model predicted 3.9998 for the input 2

Purpose of This Project

This project is designed to show:

How to build a minimal neural network in PyTorch

How to perform simple regression

How to normalize data

How to save and load model weights

How to keep a clean and readable project layout

License

Released under the MIT License.
