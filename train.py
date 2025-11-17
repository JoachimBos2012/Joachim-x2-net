# Joachim-x2-net
# This script trains a small neural network to approximate the mathematical function x².
# The model is built with PyTorch and uses supervised learning on simple numeric input/output pairs.
# After training, the network can predict squared values for new inputs.

# Import libraries
import torch
import torch.nn as nn

# Generate normalized inputs (x/100) and targets ((x²)/10000) for stable neural network training
x = torch.tensor([[i/100] for i in range(1, 201)], dtype=torch.float32) 
y = torch.tensor([[(i**2)/10000] for i in range(1, 201)], dtype=torch.float32)

# Neural network architecture: a small feedforward model that maps input values to their squared outputs
model = nn.Sequential(
    nn.Linear(1, 48), # Hidden layer is required because x² is a curved (nonlinear) function, not a straight line
    nn.Tanh(), # Tanh is used because it gives smooth, non-zero gradients needed to learn the curved x² function.
    nn.Linear(48, 32),
    nn.Tanh(), # 2 Hidden layers for the best result
    nn.Linear(32, 1)
)

criterion = nn.MSELoss() # Using MSELoss with SGD because this is a regression task and SGD minimizes squared error effectively.
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001) # Low learning rate keeps training stable and prevents the model from overshooting the x² curve.

# model.load_state_dict(torch.load("model.pth")) # unhashtag if you have already trained the model and want to train it further

# Training loop: iteratively update weights so the network learns to approximate x².
epochs = 45000 # 45000 epochs for stable training
for epoch in range(epochs):
    optimizer.zero_grad() # Reset accumulated gradients from previous step
    pred = model(x) # Forward pass: compute network prediction
    loss = criterion(pred, y) # Compute loss between prediction and target
    loss.backward() # Backpropagate gradients through the network
    optimizer.step() # Update model weights with SGD

    if epoch % 100 == 0:
        print("Epoch: ", epoch, "Loss: ", loss.item()) # Log training progress every 100 steps

torch.save(model.state_dict(), "model.pth") # saving model for predict.py and giving the opportunity to load the model in for further training
input_value = torch.tensor([[2.0]])
output = model(input_value)
print(f"The model predicted: {output.item():.4f} for the input {input_value.item()}")
