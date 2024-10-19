import numpy as np

class AdalineGD(object):
    """ADAptive Linear Neuron classifier
    
    parameters
    η - eta - learning rate (0 to 1.0)
    epochs - passes over the training dataset
    random_state - random number generator seed

    attributes
    w_ - weights after fitting
    cost_ = sum of squares cost function value in each epoch
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

        #for i in range(self.epochs):
        # calculate linear combination/dot product between examples X
        net_input = self.net_input(X)
        # run activation function (in this example - it's just an identity function)
        output = self.activation(net_input)
        # calculate errors by subtracting dot product from known labels
        errors = (y - output)
        # update weights
        self.w_[1:] += self.eta * X.T.dot(errors) # X.T.dot(errors) - matrix product
        self.w_[0] += self.eta * errors.sum()
        cost = (errors**2).sum() / 2.0
        self.cost_.append(cost)
        
        return self


    def net_input(self, X):
        """
        dot product between X (current example) and weights, enlarged by bias unit
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]


    def activation(self, X):
        """
        activation function, which here is just an identity function
        """
        return X
    

    def predct(self, X):
        """
        Threshold funtion.

        If product >= 0, return 1, -1 otherwise
        """
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
