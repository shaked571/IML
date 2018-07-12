
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
        self.misclassification = misclassification

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
        self.root = self.CART(X, y, self.get_unique_sample(X), 0)

    @staticmethod
    def get_unique_sample(X):
        A = np.array([np.unique(X[:, i]) for i in range(X.shape[1])])
        return A

    @staticmethod
    def get_error_rate(label, dim, threshold, X, y):
        """
        count the error of labeling a section with a cetain tag
        :param label: the label
        :param dim: the dimension of X to split
        :param threshold: the threshold to which we split the data
        :param X: the data
        :param y: the "true" label
        :return: the error rate
        """
        error_rate = 0
        for sample_idx in range(len(X)):
            if X[sample_idx][dim] <= threshold:
                if y[sample_idx] != label:
                    error_rate += 1
            elif y[sample_idx] != -label:
                error_rate += 1
        return error_rate / len(X)

    def split_tree(self,X, y, A):
        """
        calculate the best split for the current section of the data
        """
        min_error = 1
        min_label = 1
        min_dim = 0
        min_threshold = 0
        for dim in range(len(A)):
            for threshold in A[dim]:
                curr_error = DecisionTree.get_error_rate(1, dim, threshold, X, y)
                min_dim, min_error, min_label, min_threshold = self.get_node_param(curr_error, 1 - curr_error, dim,
                                                                                   min_dim, min_error, min_label,
                                                                                   min_threshold, threshold)

        return Node(False, samples=0, feature=min_dim, theta=min_threshold, label=min_label,
                    misclassification=min_error)

    def get_node_param(self, curr_error, curr_error_complementary, dim, min_dim, min_error, min_label, min_threshold,
                       threshold):
        if curr_error < min_error:
            min_error = curr_error
            min_threshold = threshold
            min_dim = dim
            min_label = 1
        if curr_error_complementary < min_error:
            min_error = curr_error_complementary
            min_threshold = threshold
            min_dim = dim
            min_label = -1
        return min_dim, min_error, min_label, min_threshold

    def CART(self, X, y, A, depth):
        """
        Gorw a decision tree with the CART method ()
        Parameters
        ----------
        X, y : sample
        A : array of d*m real features, A[j,:] row corresponds to thresholds over x_j
        depth : current depth of the tree

        Returns
        -------
        node : an instance of the class Node (can be either a root of a subtree or a leaf)
        """
        if depth == self.max_depth:
            leaf = self.split_tree(X, y, A)
            leaf.leaf = True
            return leaf
        else:
            new_node = self.split_tree(X, y, A)
            if new_node.misclassification == 0:
                new_node.leaf = True
                return new_node
            X_less = X[self.get_less_loc(X, new_node)]
            y_less = y[self.get_less_loc(X, new_node)]
            if len(X_less) == 0:
                new_node.left = Node(leaf=True, label=new_node.label)
            else:
                A_less = np.array(np.array([np.unique(X_less[:,i]) for i in range(X_less.shape[1])]))
                new_node.left = self.CART(X_less, y_less, A_less, depth + 1)
            X_great = X[X[:, new_node.feature] > new_node.theta]
            y_great = y[X[:, new_node.feature] > new_node.theta]
            if len(X_great) == 0:
                new_node.right = Node(leaf=True, label=-new_node.label)
            else:
                A_great = np.array(np.array([np.unique(X_great[:,i]) for i in range(X_great.shape[1])]))
                new_node.right = self.CART(X_great, y_great, A_great, depth + 1)
            return new_node

    def get_less_loc(self, X, new_node):
        return X[:, new_node.feature] <= new_node.theta

    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """
        y_pred = []
        for x in X:
            curr_node = self.root
            while not curr_node.leaf:
                curr_node = self.get_leaf(curr_node, x)
            if curr_node == self.root:
                if x[curr_node.feature] <= self.root.theta:
                    y_pred.append(curr_node.label)
                else:
                    y_pred.append(-curr_node.label)
            else:
                y_pred.append(curr_node.label)
        return np.array(y_pred)

    @staticmethod
    def get_leaf(curr_node, x):
        if x[curr_node.feature] <= curr_node.theta:
            curr_node = curr_node.left
        else:
            curr_node = curr_node.right
        return curr_node

    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
        y_hat = self.predict(X)
        y = np.array(y)
        return sum(y[i] != y_hat[i] for i in range(len(y))) / len(y)
