import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
from public_tests import *
from autils import *
import matplotlib.pyplot as plt
import logging
from lab_utils_softmax import plt_softmax
import warnings

plt.style.use('./deeplearning.mplstyle')
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
np.set_printoptions(precision=2)
warnings.simplefilter(action='ignore', category=FutureWarning)


def my_softmax(z):
    """ Softmax converts a vector of values to a probability distribution.
    Args:
      z (ndarray (N,))  : input data, N features
    Returns:
      a (ndarray (N,))  : softmax of z
    """
    a = np.exp(z) / sum(np.exp(z))
    return a


def solution():
    """
    A neural network to recognize ten handwritten digits, 0-9. This is a multiclass classification task where one of
    n choices is selected.
    The data set contains 5000 training examples of handwritten digits  1 .

    Each training example is a 20-pixel x 20-pixel grayscale image of the digit.
    Each pixel is represented by a floating-point number indicating the grayscale intensity at that location.
    The 20 by 20 grid of pixels is “unrolled” into a 400-dimensional vector.
    Each training examples becomes a single row in our data matrix X.
    This gives us a 5000 x 400 matrix X where every row is a training example of a handwritten digit image.
    The second part of the training set is a 5000 x 1 dimensional vector y that contains labels for the training set
    y = 0 if the image is of the digit 0, y = 4 if the image is of the digit 4 and so on.
    """
    # plt_act_trio()
    # z = np.array([1., 2., 3., 4.])
    # a = my_softmax(z)
    # atf = tf.nn.softmax(z)
    # print(f"My softmax result: {a}")
    # print(f"Tensorflow's softmax result: {atf}")
    # # test_my_softmax(my_softmax)
    # plt.close("all")
    # plt_softmax(my_softmax)

    X, y = load_data()  # loading the data
    # print('The shape of X is: ' + str(X.shape))
    # print('The shape of y is: ' + str(y.shape))
    m, n = X.shape

    # # Visualizing some of the randomly picked data
    # fig, axes = plt.subplots(8, 8, figsize=(5, 5))
    # fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.91])  # [left, bottom, right, top]
    #
    # # fig.tight_layout(pad=0.5)
    # widgvis(fig)
    # for i, ax in enumerate(axes.flat):
    #     # Select random indices
    #     random_index = np.random.randint(m)
    #
    #     # Select rows corresponding to the random indices and
    #     # reshape the image
    #     X_random_reshaped = X[random_index].reshape((20, 20)).T
    #
    #     # Display the image
    #     ax.imshow(X_random_reshaped, cmap='gray')
    #
    #     # Display the label above the image
    #     ax.set_title(y[random_index, 0])
    #     ax.set_axis_off()
    #     fig.suptitle("Label, image", fontsize=14)
    # plt.show()

    tf.random.set_seed(1234)  # for consistent results

    # Using Keras Sequential model and Dense Layer with a ReLU activation to construct a three layer network
    model = Sequential(
        [
            tf.keras.Input(shape=(400,)),  # specify input shape
            Dense(units=25, activation="relu", name="L1"),
            Dense(units=15, activation="relu", name="L2"),
            Dense(units=10, activation="linear", name="L3")
        ], name="my_model"
    )

    # To verify the shapes of weights and biases coming out of each layer of the network
    [layer1, layer2, layer3] = model.layers
    W1, b1 = layer1.get_weights()
    W2, b2 = layer2.get_weights()
    W3, b3 = layer3.get_weights()

    print(f"W1 shape: {W1.shape}, b1 shape: {b1.shape}")
    print(f"W2 shape: {W2.shape}, b2 shape: {b2.shape}")
    print(f"W3 shape: {W3.shape}, b3 shape: {b3.shape}")
    # test_model(model, 10, 400)

    # This defines a loss function, SparseCategoricalCrossentropy and indicates the softmax should be included with
    # the loss calculation by adding from_logits=True)
    # Also defines an optimizer called Adam with a default value for learning rate alpha
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    )
    # model.summary()

    # fitting a model to the data
    history = model.fit(
        X, y,
        epochs=40
    )
    # plot_loss_tf(history)  # to plot the loss (cost)

    # Verifying the model / prediction
    image_of_two = X[1015]  # X[1015] is an image of the digit 2
    # display_digit(image_of_two) # to display the image

    # PREDICTION
    prediction = model.predict(image_of_two.reshape(1, 400)) # reshaping it because that's how predict method wants it

    print(f" Predicting a Two: \n{prediction}")
    print(f" Largest Prediction index: {np.argmax(prediction)}")  # Index of the highest value in the array

    # The largest output is prediction[2], indicating the predicted digit is a '2'.
    # If the problem only requires a selection, that is sufficient. Use NumPy argmax to select it.
    # If the problem requires a probability, a softmax is required:
    # S0 using softmax on the output
    prediction_p = tf.nn.softmax(prediction)
    print(f"Predicting a two, probability vector: {prediction_p}")

    # Index of the highest value in the array
    yhat = np.argmax(prediction_p)
    print(f"Prediction is: {yhat}")

    # Comparing the predictions vs the labels for a random sample of 64 digits
    fig, axes = plt.subplots(8, 8, figsize=(5, 5))
    fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.91])  # [left, bottom, right, top]
    widgvis(fig)
    for i, ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)

        # Select rows corresponding to the random indices and
        # reshape the image
        X_random_reshaped = X[random_index].reshape((20, 20)).T

        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')

        # Predict using the Neural Network
        prediction = model.predict(X[random_index].reshape(1, 400))
        prediction_p = tf.nn.softmax(prediction)
        yhat = np.argmax(prediction_p)

        # Display the label above the image
        ax.set_title(f"{y[random_index, 0]},{yhat}", fontsize=10)
        ax.set_axis_off()
    fig.suptitle("Label, yhat", fontsize=14)
    plt.show()

    # Let's look at some of the errors.
    print(f"{display_errors(model, X, y)} errors out of {len(X)} images")


if __name__ == "__main__":
    solution()
