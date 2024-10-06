import numpy as np

class Perceptron:
    '''
    Perceptron is a binary classifier
    
    Basic concepts and building blocks of a perceptron:
    
    X - data set for training
    y - set of known labels for each x (observation) in X (set of observations)

    weights - one, reusable set of weights, initialized with small random values (or with zeroes), weights[0] = bias unit

    eta - learning rate, a constant between 0.0 and 1.0

    weighted sum - linear combination/dot product of the input vector xi and weight vector. For oevery observation (x in X)
                   we calculate a dot product/linear combination of the x vector and a vector of weights. We need to add
                   bias unit - weights[0] to the result of a product

    unit step function - aka activation function. Responsible for making the final decision based on the weighted sum of the inputs.
                   In this particular implementation, the step function checks wether the linear product of weights and x vector
                   is >= 0, if so, it returns 1 and -1 otherwise

    update = eta * (target_label - predicted_label) -> if (target_label - predicted_label) == 0, prediction was good, and nothing happens
             otherwise, prediction was wrong and weights are updated. If (target_label - predicted_label) != 0.0 it means that the model
             was wrong with its prediction. It means we need to update weights.

             Below, wight_ and xi are np arrays, so calculations are vectorized

             self.weight_[1:] += update * xi means that to all weights from 1st to the end, we add values in a vector xi multiplied by update
    '''
    def __init__(self, eta=0.1, epochs=50, random_state=1):
        # a constant learning rate
        self.eta=eta

        # number of iterations for the fitting algorithm
        self.epochs=epochs
        self.random_state=random_state

    
    def fit(self, X, y):
        # initialize weights with small, random values
        self.weight_=np.random.normal(loc=0.0, scale=0.1, size=1+X.shape[1])

        self.errors_ = []

        # for each epoch...
        for _ in range(self.epochs):
            errors = 0
            # for each pair of observations and known labels
            for xi, target in zip(X, y):
                # calculate update factor with the following formula
                update = self.eta * (target - self.predict(xi))
                self.weight_[1:] += update * xi # if update = 0, nothing happens
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    

    def predict(self, X):
        '''
        Activation function/unit step function

        if the value returned from net_input (linear combination) >= 0 return 1, iotherwise return -1
        '''
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    

    def net_input(self, X):
        '''
        linear combination between a single observation and weights vector, with bias unit added
        '''
        return np.dot(X, self.weight_[1:]) + self.weight_[0]