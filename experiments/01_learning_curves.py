
"""
Experiment 01: Learning curves for MNIST.

Study the evolution of the training cost and test accuracy
during stochastic gradient descent.
"""

import numpy as np
import matplotlib.pyplot as plt

# this sets the font to a LaTeX style
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import mnist_loader
import network


# ------------------------------------------
# 1. Load MNIST data
# ------------------------------------------

training_data, validation_data, test_data = (
    mnist_loader.load_data_wrapper()
)

print("MNIST dataset loaded successfully!")


# ------------------------------------------
# 2. Initialize neural network
# ------------------------------------------

N = 30 # width of the network 

net = network.Network([784, N, 10])

epochs = 30
mini_batch_size = 10
eta = 3.0


# ------------------------------------------
# 3. Train the network
# ------------------------------------------

training_cost, test_accuracy = net.SGD(
    training_data,
    epochs = epochs,
    mini_batch_size = mini_batch_size,
    eta = eta,
    test_data = test_data
)

# ------------------------------------------
# 4. Plot learning curves
# ------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

epochs_range = np.arange(1, epochs + 1)

# Training cost
axes[0].plot(epochs_range, training_cost)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Training cost")
axes[0].set_title("Training Cost")
axes[0].grid(True, alpha=0.3)

# Test accuracy
axes[1].plot(epochs_range, 100 * np.array(test_accuracy))
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Test Accuracy")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# as expected, the training cost is reduced at each epoch and the accuracy increases. 