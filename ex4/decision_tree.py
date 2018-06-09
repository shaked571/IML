"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the decision tree classifier with real-values features.
Training algorithm: CART

Author: Noga Zaslavsky
Edited: Yoav Wald, May 2018

"""
import numpy as np

class Node(object):
    """ A node in a real-valued decision tree.
        Set all the attributes properly for viewing the tree after training.
    """
    def __init__(self, leaf=True, left=None, right=None, samples=0, feature=None, theta=0.5, misclassification=0,label=None):
        """
        Parameters
        ----------
        leaf : True if the node is a leaf, False otherwise
        left : left child
        right : right child
        samples : number of training samples that got to this node
        feature : a coordinate j in [d], where d is the dimension of x (only for internal nodes)
        theta : threshold over self.feature (only for internal nodes)
        label : the label of the node, if it is a leaf
        """
        self.leaf = leaf
        self.left = left
        self.right = right
        self.samples = samples
        self.feature = feature
        self.theta = theta
        self.label = label

class DecisionTree(object):
    """ A decision tree for binary classification.
        max_depth - the maximum depth allowed for a node in this tree.
        Training method: CART
    """

    def __init__(self, max_depth):
        self.root = None
        self.max_depth = max_depth

    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """
        A = np.array([np.unique(X[:,i]) for i in range(X.shape[1])])
        self.root = self.CART(X, y, A, 0)

    @staticmethod
    def count_in_region(tag, dim, threshold, X, y):
        """
        count the error of labeling a section with a certain tag
        :param tag: the label
        :param dim: the dimension of X to split
        :param threshold: the threshold to which we split the data
        :param X: the data
        :param y: the "true" label
        :return: the error rate
        """
        errors = 0
        for sample_idx in range(len(X)):
            if X[sample_idx][dim] <= threshold:
                if y[sample_idx] != tag:
                    errors += 1
            elif y[sample_idx] != -tag:
                errors += 1
        return errors / len(X)

    def split_tree(self, X, y, A):
        """
        calculate the best split for the current section of the data
        """
        min_error = 1
        min_threshold = 0
        min_dim = 0
        min_label = 1
        for dim in range(len(A)):
            for threshold in A[dim]:
                curr_error = DecisionTree.count_in_region(1, dim, threshold, X, y)
                curr_error_minus1 = 1 - curr_error
                if curr_error < min_error:
                    min_error = curr_error
                    min_threshold = threshold
                    min_dim = dim
                    min_label = 1
                if curr_error_minus1 < min_error:
                    min_error = curr_error_minus1
                    min_threshold = threshold
                    min_dim = dim
                    min_label = -1

        return Node(False, samples=0, feature=min_dim, theta=min_threshold, label=min_label,
                    misclassification=min_error)



    def CART(self, X, y, A, depth):
        """
        Grow a decision tree with the CART method ()
        Parameters
        ----------
        X, y : sample
        A : array of d*m real features, A[j,:] row corresponds to thresholds over x_j
        depth : current depth of the tree

        Returns
        -------
        node : an instance of the class Node (can be either a root of a subtree or a leaf)
        """




    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """


    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
