# Neural Networks for Handwritten Digit Recognition, Binary

A neural network to recognize two handwritten digits, zero and one. This is a binary classification task. Automated handwritten digit recognition is widely used today - from recognizing zip codes (postal codes) on mail envelopes to recognizing amounts written on bank checks. This will be extended to recognize all 10 digits (0-9) later.



# Dataset

The data set contains 1000 training examples of handwritten digits  1 , here limited to zero and one.

Each training example is a 20-pixel x 20-pixel grayscale image of the digit.
Each pixel is represented by a floating-point number indicating the grayscale intensity at that location.
The 20 by 20 grid of pixels is “unrolled” into a 400-dimensional vector.
Each training example becomes a single row in our data matrix X.
This gives us a 1000 x 400 matrix X where every row is a training example of a handwritten digit image.
 
The second part of the training set is a 1000 x 1 dimensional vector y that contains labels for the training set
y = 0 if the image is of the digit 0, y = 1 if the image is of the digit 1.
