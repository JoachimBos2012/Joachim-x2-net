# predict.py
# This code is used to let the model predict the square of the input

# Import libraries
import torch
import torch.nn as nn

# Neural network architecture
model = nn.Sequential(
    nn.Linear(1, 48),
    nn.Tanh(),
    nn.Linear(48, 32),
    nn.Tanh(),
    nn.Linear(32, 1)
)

# Loading in the model so it can predict
model.load_state_dict(torch.load("model.pth"))
model.eval() # Set the model to evaluation mode

# Prediction
input_value  = torch.tensor([[2.0]]) # Input number the model should predict; you can freely change this value
output = model(input_value)
print(f"The model predicted: {output.item():.4f} for the input {input_value.item()}")
