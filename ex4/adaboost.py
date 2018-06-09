"""
===================================================
     Introduction to Machine Learning (67577)
===================================================
"""
import numpy as np

class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None]*T     # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """
        sample_num = X.shape[0]
        distribution = np.full(sample_num, 1/sample_num)
        for i in range(self.T):
            self.h[i] = self.WL(distribution, X, y)
            error_t = np.dot(distribution, (y != self.h[i].predict(X)))
            self.w[i] = self.get_weight(error_t)
            y_predicted = self.h[i].predict(X)
            distribution = self.get_distribution(distribution, i, y, y_predicted)

    def get_weight(self, error_t):
        return 0.5 * np.log((1 / error_t) - 1)

    def get_distribution(self, distribution, i, y, y_predicted):
        return (distribution * np.exp(-self.w[i] * y * y_predicted)) / (
        distribution * np.exp(-self.w[i] * y * y_predicted)).sum()

    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """
        return np.sign(np.array([np.dot(self.w[t], self.h[t].predict(X)) for t in range(self.T)]).sum(axis=0))

    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
        y_hat = self.predict(X)
        y = np.array(y)
        return sum(y[i] != y_hat[i] for i in range(len(y))) / len(y)
