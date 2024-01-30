import numpy as np

class Perceptron:

    # init takes input size and learning rate as arguments
    def __init__(self, N, threshold = 0, alpha=0.1):
        # initialize the weights randomly
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.threshold = threshold
        self.alpha = alpha
    
    def _step(self, x):
        return 1 if x > self.threshold else 0
    
    def fit(self, X, Y, epochs=10):
        """
        Fit the inputs X to the output Y and train the weights over the specified epochs
        """

        # insert a column of 1's as the last entry in the feature
        # Treat the bias as a trainable parameter within the weight matrix
        X = np.c_[X, np.ones((X.shape[0]))]

        # loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
            print(f"Epoch: {epoch} W: {self.W}")
            # loop over each individual data point
            for (x, target) in zip(X, Y):

                # make a prediction based on the current weights
                p = self._step(np.dot(x, self.W))

                # only perform a weight update if our prediction does not match the target
                if p != target:
                    # determine the error
                    error = p - target

                    # update the weights
                    self.W += -self.alpha * error * x

    def predict(self, X, addBias=True):
        X = np.atleast_2d(X)

        if addBias:
            X = np.c_[X, np.ones((X.shape[0]))]
        
        # take the dot product between the input features and the weights
        return self._step(np.dot(X, self.W))
    
    