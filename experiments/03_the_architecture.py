"""
Experiment 03: Studying how the architecture affects the learning.

In particular we study how the width affects learning, trying for 
different values for N, as well as how the depth affects too. 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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
# 2. Study of the width 
# ------------------------------------------

# N = [10, 20, 30, 50, 100, 200, 500]

# epochs = 30
# mini_batch_size = 10
# eta = 3.0

# training_cost_list = []
# test_accuracy_list = []
# num_parameters_list = []

# for width in N:

#     print(f"\nTraining network with {width} hidden neurons")

#     net_width = network.Network([784, width, 10])

#     training_cost, test_accuracy = net_width.SGD(
#         training_data,
#         epochs= epochs,
#         mini_batch_size = mini_batch_size,
#         eta = eta,
#         test_data = test_data
#     )

#     training_cost_list.append(training_cost[-1])
#     test_accuracy_list.append(test_accuracy[-1] * 100)

#     parameters = sum(w.size for w in net_width.weights) + sum(b.size for b in net_width.biases)
#     num_parameters_list.append(parameters)


# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# axes[0].plot(N, training_cost_list, "o-")
# axes[0].set_xlabel("Number of hidden neurons")
# axes[0].set_ylabel("Final training cost")
# axes[0].set_title("Network Width vs. Training Cost")

# axes[1].plot(N, test_accuracy_list, "o-")
# axes[1].set_xlabel("Number of hidden neurons")
# axes[1].set_ylabel("Final test accuracy (%)")
# axes[1].set_title("Network Width vs. Test Accuracy")

# plt.tight_layout()
# plt.show()


# ------------------------------------------
# 3. Network depth experiment
# ------------------------------------------

architectures = [
    [784, 30, 10],
    [784, 30, 30, 10],
    [784, 30, 30, 30, 10],
    [784, 30, 30, 30, 30, 10],
    [784, 30, 30, 30, 30, 30, 10],
    [784, 30, 30, 30, 30, 30, 30, 10],
    [784, 30, 30, 30, 30, 30, 30, 30, 10],
    [784, 30, 30, 30, 30, 30, 30, 30, 30, 10],
    [784, 30, 30, 30, 30, 30, 30, 30, 30, 30, 10]
]

epochs = 30
mini_batch_size = 10
eta = 3.0

training_cost_list = []
test_accuracy_list = []
num_parameters_list = []
depths = []

for architecture in architectures:

    print(f"\nTraining architecture: {architecture}")

    net_depth = network.Network(architecture)

    training_cost, test_accuracy = net_depth.SGD(
        training_data,
        epochs=epochs,
        mini_batch_size=mini_batch_size,
        eta=eta,
        test_data=test_data
    )

    training_cost_list.append(training_cost[-1])
    test_accuracy_list.append(test_accuracy[-1] * 100)

    parameters = sum(w.size for w in net_depth.weights) + sum(b.size for b in net_depth.biases)
    num_parameters_list.append(parameters)

    depths.append(len(architecture) - 2)

# Plot results

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(depths, training_cost_list, "o-")
axes[0].set_xlabel("Number of hidden layers")
axes[0].set_ylabel("Final training cost")
axes[0].set_title("Network Depth vs. Training Cost")
axes[0].set_xticks(depths)

axes[1].plot(depths, test_accuracy_list, "o-")
axes[1].set_xlabel("Number of hidden layers")
axes[1].set_ylabel("Final test accuracy (%)")
axes[1].set_title("Network Depth vs. Test Accuracy")
axes[1].set_xticks(depths)

plt.tight_layout()
plt.show()