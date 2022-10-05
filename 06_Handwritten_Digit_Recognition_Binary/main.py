import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from autils import *
from public_tests import *
import logging
import warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
warnings.simplefilter(action='ignore', category=FutureWarning)


def solution():
    """
    Neural network to recognize two handwritten digits, zero and one. This is a binary classification task.

    The data set contains 1000 training examples of handwritten digits  1 , here limited to zero and one.
    Each training example is a 20-pixel x 20-pixel grayscale image of the digit.
    Each pixel is represented by a floating-point number indicating the grayscale intensity at that location.
    The 20 by 20 grid of pixels is “unrolled” into a 400-dimensional vector.
    Each training example becomes a single row in our data matrix X.
    This gives us a 1000 x 400 matrix X where every row is a training example of a handwritten digit image.
    The second part of the training set is a 1000 x 1 dimensional vector y that contains labels for the training set
    y = 0 if the image is of the digit 0, y = 1 if the image is of the digit 1.
    """
    X, y = load_data()

    m, n = X.shape

    # visualising some of the data picked randomly
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    fig.tight_layout(pad=0.1)
    for i, ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)

        # Select rows corresponding to the random indices and reshape the image
        X_random_reshaped = X[random_index].reshape((20, 20)).T

        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')

        # Display the label above the image
        ax.set_title(y[random_index, 0])
        ax.set_axis_off()
    # plt.show()

    # THE MODEL
    model = Sequential(
                [
                    tf.keras.Input(shape=(400,)),
                    Dense(units=25, activation='sigmoid', name='Layer1'), # first layer with 25 neurons
                    Dense(units=15, activation='sigmoid', name='Layer2'), # second layer with 15 neurons
                    Dense(units=1, activation='sigmoid', name='Layer3')   # final layer with 1 neuron
                 ], name="my_model"
            )

    # model.summary()  # to get a summary of the parameters
    # test_c1(model)   # to test the model

    # pulling the layers to examine the shapes of weights and biases set by the model
    [layer1, layer2, layer3] = model.layers
    W1, b1 = layer1.get_weights()
    W2, b2 = layer2.get_weights()
    W3, b3 = layer3.get_weights()

    print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
    print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
    print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")

    # print(model.layers[2].get_weights())

    # defining a loss function and run gradient descent to fit the weights of the model to the training data
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(0.001)
    )

    model.fit(
        X, y,
        epochs=20
    )

    # predicting for some of the data whose output we already know just to verify
    # prediction = model.predict(X[0].reshape(1, 400)) # X[0] is a zero, so predicting 0
    # print(f"predicting a zero: {prediction}")
    # # prediction = model.predict(X[500].reshape(1, 400))  # X[0] is a one, so predicting 1
    # # print(f"predicting a one: {prediction}")
    #
    # setting a threshold value to show the output based on the probability given by the model
    # if prediction >= 0.5:
    #     yhat = 1
    # else:
    #     yhat = 0
    #
    # print(f"prediction after applying threshold: {yhat}")

    # comparing the predictions vs the labels for a random sample of 64 digits
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    fig.tight_layout(pad=0.1, rect=[0, 0.03, 1, 0.92])  # [left, bottom, right, top]

    for i, ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)

        # Select rows corresponding to the random indices and reshape the image
        X_random_reshaped = X[random_index].reshape((20, 20)).T

        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')

        # Predict using the Neural Network
        prediction = model.predict(X[random_index].reshape(1, 400))
        if prediction >= 0.5:
            yhat = 1
        else:
            yhat = 0

        # Display the label above the image
        ax.set_title(f"{y[random_index, 0]},{yhat}")
        ax.set_axis_off()
    fig.suptitle("Label, yhat", fontsize=16)
    plt.show()


if __name__ == "__main__":
    solution()
