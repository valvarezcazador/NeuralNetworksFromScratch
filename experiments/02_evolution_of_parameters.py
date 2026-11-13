"""
Experiment 02: Studying how the parameters evolve.

Study the evolution of the weights and biases at each 
epoch during training, as well as how their distribution
changes along training. Recall that the 
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
# 2. Initialize neural network
# ------------------------------------------

N = 30

net = network.Network([784, N, 10])

epochs = 30
mini_batch_size = 10
eta = 3.0


# ---------------------------------------------------
# 3. Train the network: evolution of parameters
# ---------------------------------------------------

training_cost, test_accuracy = net.SGD(
    training_data,
    epochs = epochs,
    mini_batch_size = mini_batch_size,
    eta = eta,
    test_data = test_data
)

weights_history = net.weights_history #obtaining the atribute weights after training, it is stored at the end of each training epoch
biases_history = net.biases_history #obtaining the atribute biases after training, it is stored at the end of each training epoch 

# print(len(weights_history))       # 31 : weights at initialization + weights after 
# print(weights_history[0][0].shape) # (30, 784) : first later 784 ---> hidden layer 30 
# print(weights_history[0][1].shape) # (10, 30) : hidden layer 30 ---> output layer 10

# now we will plot the evolution of W^(2) \in R^{30x784}, W^(3) \in R^{30x10}, and b^(2) \in  R^30 
# for that we will do a colourmap of W^(2), W^(3) and b^(2) every 5 epochs, starting from the 

# ------------------------------------------
# Evolution of weights and biases
# ------------------------------------------

selected_epochs = [0, 1, 5, 10, 15, 20, 25, 30]

parameters = [
    (weights_history, 0, r"First-Layer Weights $W^{(2)}$ ", False),
    (weights_history, 1, r"Second-Layer Weights $W^{(3)}$", False),
    (biases_history, 0, r"Hidden-Layer Biases $b^{(2)}$", True),
    (biases_history, 1, r"Output-Layer Biases $b^{(3)}$", True)
]

for history, index, title, transpose in parameters:

    # Common colour scale across epochs
    vmax = max(
        np.max(np.abs(history[t][index]))
        for t in selected_epochs
    )

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))

    for ax, t in zip(axes.flat, selected_epochs):

        parameter = history[t][index]

        if transpose:
            parameter = parameter.T

        im = ax.imshow(
            parameter,
            cmap="coolwarm",
            aspect="auto",
            vmin=-vmax,
            vmax=vmax
        )

        ax.set_title("Initial" if t == 0 else f"Epoch {t}")
        ax.axis("off")

    # Common colour bar
    fig.colorbar(
        im,
        ax = axes.ravel().tolist(),
        orientation = "vertical",
        fraction = 0.02,
        pad = 0.02,
        label = "Parameter value"
    )

    fig.suptitle(f"Evolution of {title}")
    plt.show()


#----------------------------------------------------------------------------------------------------------
# 4. Study the distribution of the weights and biases, which have been initially sampled from a N(0,1)
#----------------------------------------------------------------------------------------------------------

# At each epoch we have a sample of W^(2), W^(3), b^(2), b^(3). Is Gaussianity preserved? Recall that everything is originally sampled from N(0,1) 
# Let's study again the distribution of the parameters on the selected epochs 

for history, index, title, transpose in parameters:

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))

    for ax, t in zip(axes.flat, selected_epochs):

        data = history[t][index].ravel()

        # Fit a Gaussian distribution
        mu, sigma = stats.norm.fit(data)

        # Plot empirical distribution
        ax.hist(
            data,
            bins = 40,
            density = True,
            alpha = 0.6,
            label = "Empirical distribution"
        )

        # Plot fitted Gaussian
        x = np.linspace(data.min(), data.max(), 300)
        ax.plot(
            x,
            stats.norm.pdf(x, mu, sigma),
            linewidth = 2,
            label = "Gaussian fit"
        )

         # Plot N(0,1) Gaussian
        x = np.linspace(data.min(), data.max(), 300)
        ax.plot(
            x,
            stats.norm.pdf(x, 0, 1),
            linewidth = 2,
            label = r"$N(0,1)$"
            )

        ax.set_title("Initial" if t == 0 else f"Epoch {t}")

    axes[0, 0].legend(fontsize = 8)
    fig.suptitle(f"Parameter Distributions: {title}", fontsize = 14)
    plt.tight_layout()
    plt.show()