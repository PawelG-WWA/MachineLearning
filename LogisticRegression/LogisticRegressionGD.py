import numpy as np

class LogisticRegressionGD(object):
    """ADAptive Linear Neuron classifier
    
    parameters
    η - eta - learning rate (0 to 1.0)
    epochs - passes over the training dataset
    random_state - random number generator seed

    attributes
    w_ - weights after fitting
    cost_ = logistic cost function value in each epoch
    """
    def __init__(self, eta=0.01, epochs=50, random_state=1):
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state


    def fit(self, X, y):
        """
        X - training vectors with n examples and m p features
        y - target values - len(y) == len(X) (each example has a corresponding target value)
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.cost_ = []

        for i in range(self.epochs):
            # calculate linear combination/dot product between examples X
            net_input = self.net_input(X)
            # run activation function
            output = self.activation(net_input)
            # calculate errors by subtracting dot product from known labels
            errors = (y - output)
            # update weights
            self.w_[1:] += self.eta * X.T.dot(errors) # X.T.dot(errors) - matrix product
            self.w_[0] += self.eta * errors.sum()
            cost = -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))
            self.cost_.append(cost)
        
        return self


    def net_input(self, X):
        """
        dot product between X (all examples) and weights, enlarged by bias unit
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]


    def activation(self, z):
        """
        Compute logistic sigmoid activation
        """
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
    

    def predct(self, X):
        """
        Threshold funtion.

        If product >= 0, return 1, -1 otherwise
        """
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
