"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for classification with Bagging.

Author: Yoav Wald

"""
import numpy as np


class Bagging(object):

    def __init__(self, L, B, size_T):
        """
        Parameters
        ----------
        L : the class of the base learner
        T : the number of base learners to learn
        """
        self.L = L
        self.B = B
        self.size_T = size_T
        self.h = [None]*B

    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """
        m, d = X.shape
        for b in range(self.B):
            S_tag_idx = np.random.choice(np.array(range(m)), m)
            S_tag = X[S_tag_idx]
            y_tag = y[S_tag_idx]
            self.h[b] = self.L(self.size_T)
            self.h[b].train(S_tag, y_tag)

    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """
        return np.sign(np.array([(1/self.B) * self.h[t].predict(X) for t in range(self.B)]).sum(axis=0))


    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
        y_hat = self.predict(X)
        y = np.array(y)
        ret_error_numerator= sum(y[i] != y_hat[i] for i in range(len(y)))
        ret_error_denominator = len(y)
        ret_error = ret_error_numerator / ret_error_denominator
        return ret_error
