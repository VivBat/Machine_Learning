import numpy as np
from test_utils import *
from utils import *
import matplotlib.pyplot as plt
from public_tests import *


def sigmoid(z):
    """
    Compute the sigmoid of z
    Args:
        z (ndarray): A scalar, numpy array of any size.
    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
    """
    return 1 / (1 + np.exp(-z))


def compute_cost_reg(x, y, w, b, lambda_=1):
    """
    Computes the cost over all examples
    Args:
      x : (ndarray Shape (m,n)) data, m examples by n features
      y : (array_like Shape (m,)) target value
      w : (array_like Shape (n,)) Values of parameters of the model
      b : scalar Values of bias parameter of the model
      lambda_ : regularisation parameter
    Returns:
      total_cost: (scalar)         cost
    """
    m = x.shape[0]

    z = (w * x).sum(axis=1) + b             # an array of shape (m,1)
    loss = -y * np.log(sigmoid(z)) - (1 - y) * np.log(1 - sigmoid(z))   # an array of shape (m,1)
    reg_term = (lambda_ / (2*m)) * sum(w**2)
    cost = sum(loss) / m + reg_term  # the array above turned into a scalar

    # # solution with a loop
    # cost = 0
    # for i in range(m):
    #     z = np.dot(w, x[i]) + b
    #     loss = -y[i] * np.log(sigmoid(z)) - (1 - y[i]) * np.log(1 - sigmoid(z))
    #     cost += loss
    # cost = cost/m

    return cost


def compute_gradient_reg(x, y, w, b, lambda_=1):
    """
    Computes the gradient for logistic regression

    Args:
        x : (ndarray Shape (m,n)) variable such as house size
        y : (array_like Shape (m,1)) actual value
        w : (array_like Shape (n,1)) values of parameters of the model
        b : (scalar)                 value of parameter of the model
        lambda_ : regularisation parameter
    Returns
        dj_dw: (array_like Shape (n,1)) The gradient of the cost w.r.t. the parameters w.
        dj_db: (scalar)                The gradient of the cost w.r.t. the parameter b.
    """
    m, n = x.shape

    # Calculating without loops
    z = (w * x).sum(axis=1) + b             # an array of shape (m,1)
    dJdw = np.dot(sigmoid(z) - y, x) + lambda_*w       # an array of shape (n,1)
    dJdb = (sigmoid(z) - y).sum(axis=0)     # scalar

    # Calculating with loops, SLOWER obviously
    # dJdw = np.zeros(n)
    # dJdb = 0
    # for i in range(m):
    #     z = np.dot(w, x[i]) + b
    #     for j in range(n):
    #         dJdw[j] += (sigmoid(z) - y[i]) * x[i][j]
    #     dJdb += sigmoid(z) - y[i]

    dJdw = dJdw / m         # scaling as per the formula
    dJdb = dJdb / m         # scaling as per the formula

    return dJdb, dJdw


def gradient_descent(x, y, w, b, cost_function, gradient_function, alpha, num_iters, lambda_):
    """
        Performs batch gradient descent to learn theta. Updates theta by taking
        num_iters gradient steps with learning rate alpha

        Args:
          x :    (array_like Shape (m, n)
          y :    (array_like Shape (m,))
          w : (array_like Shape (n,))  Initial values of parameters of the model
          b : (scalar)                 Initial value of parameter of the model
          cost_function:                  function to compute cost
          gradient_function:              function to compute gradient
          alpha : (float)                 Learning rate
          num_iters : (int)               number of iterations to run gradient descent
          lambda_ :                       regularisation parameter

        Returns:
          w : (array_like Shape (n,)) Updated values of parameters of the model after
              running gradient descent
          b : (scalar)                Updated value of parameter of the model after
              running gradient descent
        """
    cost_hist = []
    for i in range(num_iters):
        dJdb, dJdw = gradient_function(x, y, w, b, lambda_)      # calculating the gradient at the current w and b
        w = w - alpha * dJdw                            # updating w
        b = b - alpha * dJdb                            # updating b
        cost = cost_function(x, y, w, b, lambda_)                # computing cost with the updated values of w and b
        cost_hist.append(cost)                          # adding to cost history

        if i % 1000 == 0:
            print(f"Iteration no. {i}: {cost}")  # prints cost for every 1000th iteration

    return w, b, cost_hist


def predict(x, w, b):
    """
        Predict whether the label is 0 or 1 using learned logistic
        regression parameters w

        Args:
        X : (ndarray Shape (m, n))
        w : (array_like Shape (n,))      Parameters of the model
        b : (scalar, float)              Parameter of the model

        Returns:
        p: (ndarray (m,1))
            The predictions for X using a threshold at 0.5
        """
    z = (w * x).sum(axis=1) + b     # an array of shape (m,1)
    prediction = sigmoid(z) > 0.5   # an array which has element = True if sigmoid(z) > 0.5, else False
    return prediction


def solution():
    x_train, y_train = load_data('data/ex2data2.txt')
    x_mapped = map_feature(x_train[:, 0], x_train[:, 1])       # mapping the 2 features to 27 features to get non linear boundary

    # plot_data(x_train, y_train, "+", "o")        # plotting the data, '+' represents y = 1, 'o' represents y = 0
    # plt.legend()
    # plt.show()

    n = x_mapped.shape[1]  # no of columns(features)
    w_init = np.zeros(n)  # initial values for w
    b_init = 1           # initial value for b
    alpha = 0.01         # learning rate
    num_iters = 10000     # no of iterations to run
    lambda_ = 0.01        # regularisation parameter

    # compute_cost_reg_test(compute_cost_reg)
    # compute_gradient_reg_test(compute_gradient_reg)

    w_final, b_final, J_hist = gradient_descent(x_mapped, y_train, w_init, b_init, compute_cost_reg, compute_gradient_reg, alpha,
                                                num_iters, lambda_)   # the final values for w and b for the model, also returns the history of cost

    p = predict(x_mapped, w_final, b_final)     # predicting output based on model to compare with the training data output
    print('Train Accuracy: %f' % (np.mean(p == y_train) * 100))  # comparing model output to training data output

    plot_decision_boundary(w_final, b_final, x_mapped, y_train)  # plots the decision boundary
    plt.plot()
    plt.show()


if __name__ == "__main__":
    solution()
