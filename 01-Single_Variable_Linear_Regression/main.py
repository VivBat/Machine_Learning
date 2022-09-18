import numpy as np
from utils import *
from public_tests import *
import matplotlib.pyplot as plt


def compute_cost(x, y, w, b):
    '''
    :param x: the features
    :param y: the output
    :param w: initial weight for x
    :param b: initial intercept for hte model
    :return: the cost
    '''

    cost = 0
    m = x.shape[0]
    for i in range(m):
        cost += (np.dot(w, x[i]) + b - y[i]) ** 2
    cost = cost / (2 * m)
    return cost


def compute_gradient(x, y, w, b):
    '''
    :param x: the features
    :param y: the output
    :param w: initial weight for x
    :param b: initial intercept for hte model
    :return: the gradient
    '''
    m = x.shape[0]
    dJdw = np.zeros(x[0].size)
    # print(dJdw)
    dJdb = 0
    for i in range(m):
        for j in range(len(dJdw)):
            if len(dJdw) < 2:
                dJdw[j] += (np.dot(w, x[i]) + b - y[i]) * x[i]
            else:
                dJdw[j] += (np.dot(w, x[i]) + b - y[i]) * x[i][j]
            dJdb += np.dot(w, x[i]) + b - y[i]
            # print(dJdw, dJdb)

    dJdw = dJdw / m
    dJdb = dJdb / m

    return dJdw, dJdb


def gradient_descent(x, y, w, b, cost_function, gradient_function, alpha, iters):
    cost_history = []
    for i in range(iters):
        dJdw, dJdb = gradient_function(x, y, w, b)
        w = w - alpha * dJdw
        b = b - alpha * dJdb
        # print(w, b)

        cost = cost_function(x, y, w, b)
        cost_history.append(cost)

        if i % 150 == 0:
            print(f"Iteration {i}: Cost: " + str(cost))

    return w, b, cost_history


def solution():
    ## without scaling features
    # load the data from the text file
    x_train, y_train = load_data()

    ## scaling features
    # x_train1, y_train1 = load_data()
    #
    # x_mu = np.mean(x_train1, axis=0)
    # x_sigma = np.std(x_train1, axis=0)
    # y_mu = np.mean(y_train1, axis=0)
    # y_sigma = np.std(y_train1, axis=0)
    # # print(x_mu, x_sigma, y_mu, y_sigma)
    #
    # x_train = (x_train1 - x_mu) / x_sigma
    # y_train = (y_train1 - y_mu) / y_sigma

    # parameters
    # number of training examples
    m = x_train.shape[0]
    w_init = np.zeros(x_train[0].size)
    b_init = 0
    alpha = 0.01
    iters = 1500

    # cost = compute_cost(x_train, y_train, w_init, b_init)
    # print(cost)
    # compute_cost_test(compute_cost)

    # grad = compute_gradient(x_train, y_train, w_init, b_init)
    # print(grad)
    # compute_gradient_test(compute_gradient)

    w_final, b_final, cost_hist = gradient_descent(x_train, y_train, w_init, b_init, compute_cost, compute_gradient,
                                                   alpha, iters)
    print(w_final, b_final)

    # based on the model, predicting the profit for a city of population 35000
    population = 3.5
    # pop_norm = (population - x_mu) / x_sigma
    prediction1 = w_final * population + b_final
    print(f"Profit is: ${prediction1 * 10000}")

    # to plot the model and the training data
    predicted = np.zeros(m)
    for i in range(m):
        predicted[i] = np.dot(w_final, x_train[i]) + b_final

    plt.plot(x_train, predicted, c="b", label='Prediction')
    plt.scatter(x_train, y_train, marker='x', c='r', label='Training data')
    plt.xlabel("x1")
    plt.ylabel("Output")
    plt.title("Linear Regression")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    solution()
