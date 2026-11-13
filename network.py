"""
This script implements an algorithm based on Machine Learning
techniques that recognizes handwritten digits using the MNIST database.

This project is my "Hello_World.py" in Machine Learning :).
"""

import numpy as np
np.random.seed(42) 

# --------------------------------------------------
# Activation functions
# --------------------------------------------------

def sigmoid(z):
    '''Implements the sigmoid activation function'''
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    '''Returns the derivative of the sigmoid function'''
    s = sigmoid(z)
    return s * (1 - s)


# --------------------------------------------------
# Neural Network class
# --------------------------------------------------

class Network():
    # the only input of the class Network is the list of sizes, which is nothing but a list whose lenght is the number of total layer (including the input) and the 
    # components are the number of neurons per layer, in our notation sizes = [n0, n1, ..., n_{l-1}, n_l, n_{l+1}, ..., n_L]. 

    def __init__(self, sizes):
        '''Initializes the neural network'''

        self.num_layers = len(sizes)
        self.sizes = sizes

        # Random initialization of biases
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # they are chosen to be N(0,1) 
        # the biases vector of each layer (except the input layer) is chosen to be (y,1), where y is the amount of neurons of that layer. The input layer does not have biases. 

        # Random initialization of weights
        self.weights = [
            np.random.randn(y, x)
            for x, y in zip(sizes[:-1], sizes[1:])
        ]
        # they are chosen to be sampled from N(0,1)
        # the input/uoutput layers do not have weights, that's why the loop iterates over sizes[:-1] to sizes[1:], simply excluding the input/output layers.
        # Since the weights from n_{l-1} to n_l arrange a matrix W^(l) whose dimension is n_l x n_{l-1}, that explains why the np.random.randn(y, x). 

        # Evolution of the parameters, store copies of the values at initialization 
        self.biases_history = [[b.copy() for b in self.biases]]
        self.weights_history = [[w.copy() for w in self.weights]]

    # --------------------------------------------------
    # Feedforward
    # --------------------------------------------------

    def feedforward(self, a):
        '''Returns the output of the network when "a" is the input'''

        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)

        return a

    # --------------------------------------------------
    # Cost function
    # --------------------------------------------------

    def cost_function(self, data): 
        '''
        Computes the average quadratic error of the dataset 'data'
        '''
        n = len(data)
        cost = 0

        for x, y in data: 
            network_output = self.feedforward(x)
            cost = cost + 0.5 * np.linalg.norm(network_output - y)**2

        return cost/n

    # --------------------------------------------------
    # Stochastic Gradient Descent
    # --------------------------------------------------

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        '''
        Implements Stochastic Gradient Descent (SGD).

        Arguments:
        - training_data: list of tuples (x, y)
        - epochs: number of training epochs
        - mini_batch_size: size of each mini-batch
        - eta: learning rate
        - test_data: optional test dataset
        '''

        n = len(training_data)
        training_cost = []
        test_accuracy = []
        for j in range(epochs):

            # Shuffle training data
            np.random.shuffle(training_data)

            # Divide training data into mini-batches
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            # Update network for each mini-batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            # Store parameters after each epoch, study how these change at the end of each epoch!!!
            self.weights_history.append([w.copy() for w in self.weights])
            self.biases_history.append([b.copy() for b in self.biases])

            # Calculate objective function
            cost = self.cost_function(training_data)
            training_cost.append(cost)

            # Evaluate network
            if test_data is not None:
                accuracy = self.evaluate(test_data) / len(test_data)
                test_accuracy.append(accuracy)

                print(
                    f"Epoch {j+1:2d}/{epochs} | "
                    f"Cost: {cost:.6f} | "
                    f"Test accuracy: {100*accuracy:.2f}%"
                )
            else:
                print(
                    f"Epoch {j+1:2d}/{epochs} | "
                    f"Cost: {cost:.6f}"
                )

        return training_cost, test_accuracy
    
    # --------------------------------------------------
    # Update mini-batch
    # --------------------------------------------------

    def update_mini_batch(self, mini_batch, eta):
        '''
        Updates the network's weights and biases using
        gradient descent and backpropagation.
        '''

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:

            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            nabla_b = [
                nb + dnb
                for nb, dnb in zip(nabla_b, delta_nabla_b)
            ]

            nabla_w = [
                nw + dnw
                for nw, dnw in zip(nabla_w, delta_nabla_w)
            ]

        # Gradient descent update
        self.weights = [
            w - (eta / len(mini_batch)) * nw
            for w, nw in zip(self.weights, nabla_w)
        ]

        self.biases = [
            b - (eta / len(mini_batch)) * nb
            for b, nb in zip(self.biases, nabla_b)
        ]

    # --------------------------------------------------
    # Backpropagation
    # --------------------------------------------------

    def backprop(self, x, y):
        '''
        Returns a tuple (nabla_b, nabla_w) representing
        the gradients of the cost function with respect
        to every bias and weight in the network.
        '''

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Feedforward

        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):

            z = np.dot(w, activation) + b
            zs.append(z)

            activation = sigmoid(z)
            activations.append(activation)

        # Backpropagation

        # Output layer error
        delta = (
            self.cost_derivative(activations[-1], y)
            * sigmoid_prime(zs[-1])
        )

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Hidden layers
        for l in range(2, self.num_layers):

            z = zs[-l]
            sp = sigmoid_prime(z)

            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp

            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_b, nabla_w

    # --------------------------------------------------
    # Cost derivative
    # --------------------------------------------------

    def cost_derivative(self, output_activations, y):
        '''
        Returns the derivative of the quadratic cost function:

        C = 1/2 * ||y - a^(L)||^2
        '''

        return output_activations - y

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    def evaluate(self, test_data):
        '''Returns the number of correctly classified test examples'''

        test_results = [
            (np.argmax(self.feedforward(x)), y)
            for (x, y) in test_data
        ]

        return sum(
            int(x == y)
            for (x, y) in test_results
        )