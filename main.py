"""
Main script: handwritten digit recognition using MNIST.
"""

import numpy as np
import mnist_loader
import network
import matplotlib.pyplot as plt

#------------------------------------------
# Loading MNIST data
#------------------------------------------

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

print("MNIST dataset loaded successfully!")
print("Training examples:", len(training_data))
print("Validation examples:", len(validation_data))
print("Test examples:", len(test_data))


#------------------------------------------
# Initialize our neural network
#------------------------------------------

N = 30 # hidden neurons, width 
net = network.Network([784, N, 10]) 

# It has 784 input neurons since the MNIST pictures are stored as 784-dimensional vectors and 10 output neurons. 


#------------------------------------------
# Training our neural network
#------------------------------------------

net.SGD(
    training_data,
    epochs = 30,
    mini_batch_size = 10,
    eta = 3.0, #learning rate
    test_data = test_data
)

#------------------------------------------
# Display 25 random test images
#------------------------------------------

fig, axes = plt.subplots(5, 5, figsize=(10, 10))

# Select 25 random test images
indices = np.random.choice(len(test_data), 25, replace=False)

for ax, index in zip(axes.flat, indices):

    x, y = test_data[index]

    # Make a prediction
    output = net.feedforward(x)
    prediction = np.argmax(output)

    # Display image
    ax.imshow(x.reshape(28, 28), cmap="gray")
    ax.set_title(f"Pred: {prediction} | True: {y}", fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.show()