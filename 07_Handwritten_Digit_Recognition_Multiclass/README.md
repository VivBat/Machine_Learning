# Neural Networks for Handwritten Digit Recognition, Multiclass

A neural network to recognize  handwritten digits (0-9). This is a multiclass classification task. Automated handwritten digit recognition is widely used today - from recognizing zip codes (postal codes) on mail envelopes to recognizing amounts written on bank checks. 


# Dataset

The data set contains 5000 training examples of handwritten digits  1 .

Each training example is a 20-pixel x 20-pixel grayscale image of the digit.
Each pixel is represented by a floating-point number indicating the grayscale intensity at that location.
The 20 by 20 grid of pixels is “unrolled” into a 400-dimensional vector.
Each training examples becomes a single row in our data matrix X.
This gives us a 5000 x 400 matrix X where every row is a training example of a handwritten digit image.
 
The second part of the training set is a 5000 x 1 dimensional vector y that contains labels for the training set
y = 0 if the image is of the digit 0, y = 4 if the image is of the digit 4 and so on.

# Model representation

The neural network has two dense layers with ReLU activations followed by an output layer with a linear activation.
Recall that our inputs are pixel values of digit images.
Since the images are of size  20×20 , this gives us  400  inputs

The parameters have dimensions that are sized for a neural network with  25  units in layer 1,  15  units in layer 2 and  10  output units in layer 3, one for each digit.

The dimensions of these parameters is determined as follows:

If network has  𝑠𝑖𝑛  units in a layer and  𝑠𝑜𝑢𝑡  units in the next layer, then
𝑊  will be of dimension  𝑠𝑖𝑛×𝑠𝑜𝑢𝑡 .
𝑏  will be a vector with  𝑠𝑜𝑢𝑡  elements
Therefore, the shapes of W, and b, are

layer1: The shape of W1 is (400, 25) and the shape of b1 is (25,)
layer2: The shape of W2 is (25, 15) and the shape of b2 is: (15,)
layer3: The shape of W3 is (15, 10) and the shape of b3 is: (10,)

# Tensorflow Model Implementation

Tensorflow models are built layer by layer. A layer's input dimensions ( 𝑠𝑖𝑛  above) are calculated for you. You specify a layer's output dimensions and this determines the next layer's input dimension. The input dimension of the first layer is derived from the size of the input data specified in the model.fit statement below.

# Softmax placement

Numerical stability is improved if the softmax is grouped with the loss function rather than the output layer during training. This has implications when building the model and using the model.

Building:
The final Dense layer should use a 'linear' activation. This is effectively no activation.
The model.compile statement will indicate this by including from_logits=True. loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
This does not impact the form of the target. In the case of SparseCategorialCrossentropy, the target is the expected digit, 0-9.

Using the model:
The outputs are not probabilities. If output probabilities are desired, apply a softmax function.

