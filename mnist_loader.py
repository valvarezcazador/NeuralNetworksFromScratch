
"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.

Based on Michael Nielsen's original mnist_loader.py,
adapted for Python 3.13.
"""

import pickle
from pathlib import Path
import numpy as np


def load_data():
    """
    Return the MNIST data as a tuple containing the
    training data, validation data and test data.
    """

    # Locate the dataset relative to this file
    path = Path(__file__).parent / "data" / "mnist.pkl"

    with open(path, "rb") as f:
        training_data, validation_data, test_data = pickle.load(
            f, encoding="latin1"
        )

    return training_data, validation_data, test_data

def load_data_wrapper():
    """
    Return a tuple containing (training_data,
    validation_data, test_data).

    Training data:
        List of 50,000 tuples (x, y).
        x: 784-dimensional input vector.
        y: 10-dimensional one-hot output vector.

    Validation and test data:
        Lists of tuples (x, y).
        x: 784-dimensional input vector.
        y: integer representing the correct digit.
    """

    tr_d, va_d, te_d = load_data()

    # Training data
    training_inputs = [
        np.reshape(x, (784, 1))
        for x in tr_d[0]
    ]

    training_results = [
        vectorized_result(y)
        for y in tr_d[1]
    ]

    # In Python 3, zip() returns an iterator.
    # We convert it into a list because SGD needs
    # to shuffle the training data.
    training_data = list(zip(training_inputs, training_results))

    # Validation data
    validation_inputs = [
        np.reshape(x, (784, 1))
        for x in va_d[0]
    ]

    validation_data = list(zip(validation_inputs, va_d[1]))

    # Test data
    test_inputs = [
        np.reshape(x, (784, 1))
        for x in te_d[0]
    ]

    test_data = list(zip(test_inputs, te_d[1]))

    return training_data, validation_data, test_data


def vectorized_result(j):
    """
    Return a 10-dimensional unit vector
    corresponding to the digit j.
    """

    e = np.zeros((10, 1))
    e[j] = 1.0

    return e