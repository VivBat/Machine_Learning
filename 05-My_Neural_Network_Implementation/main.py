# implementing forward propagation in a neural network from scratch without using any libraries

import numpy as np
import matplotlib.pyplot as plt
from autils import *
from public_tests import test_c2, test_c3
import warnings


def sigmoid(z):
    """
    Compute the sigmoid of z
    Args:
        z (ndarray): A scalar, numpy array of any size.
    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
    """
    return 1 / (1 + np.exp(-z))


def my_dense(a_in, W, b, activation):
    """
    Computes dense layer
    Args:
      a_in (ndarray (m,n)) : Data, m examples, n features each
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j,1)) : bias vector, j units
      activation:            activation function (e.g. sigmoid, relu..)
    Returns
      a_out (ndarray (m,j)) : m examples, j units
    """
    x_out_temp = np.matmul(a_in, W) + b
    x_out = activation(x_out_temp)

    return x_out


def my_sequential(x_in, W1, b1, W2, b2, W3, b3):
    """
    Creates a 3 layer ntwork
    x_in: Input to the network
    W1: Weights for first layer
    b1: Biases for first layer
    W2: Weights for second layer
    b2: Biases for second layer
    W3: Weights for third layer
    b3: Biases for third layer
    Returns : the output of the network
    """
    a1 = my_dense(x_in, W1, b1, sigmoid)
    a2 = my_dense(a1, W2, b2, sigmoid)
    a3 = my_dense(a2, W3, b3, sigmoid)

    return a3


def solution():
    x, y = load_data()

    # Visualising some data
    warnings.simplefilter(action='ignore', category=FutureWarning)
    m, n = x.shape
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    fig.tight_layout(pad=0.1)

    for i, ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)

        # Select rows corresponding to the random indices and
        # reshape the image
        X_random_reshaped = x[random_index].reshape((20, 20)).T

        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')

        # Display the label above the image
        ax.set_title(y[random_index, 0])
        ax.set_axis_off()
    plt.show()

    # test_c3(my_dense)


if __name__ == '__main__':
    solution()
