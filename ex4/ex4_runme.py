"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Running script for Ex4.

Author:
Date: May, 2018

"""

import numpy as np
from SynData import *
from ex4_tools import decision_boundaries, h_opt, DecisionStump
from matplotlib.pyplot import *
import adaboost as aba
import decision_tree as dta

X_train = np.loadtxt("SynData/X_train.txt")
X_test = np.loadtxt("SynData/X_test.txt")
X_val = np.loadtxt("SynData/X_val.txt")
y_train = np.loadtxt("SynData/y_train.txt")
y_test = np.loadtxt("SynData/y_test.txt")
y_val = np.loadtxt("SynData/y_val.txt")

def Q3(): # AdaBoost
    val_error = []
    train_error =[]
    for T in range(1, 205, 5):
        adaboost = aba.AdaBoost(DecisionStump, T)
        adaboost.train(X_train, y_train)
        train_error.append(adaboost.error(X_train, y_train))
        val_error.append(adaboost.error(X_val, y_val))

    plot(list(range(1, 205, 5)), train_error)
    plot(list(range(1, 205, 5)), val_error)
    xlabel("Iteration_num")
    ylabel("error")
    legend(["Training Error", "Validation Error"], loc=5)
    show()

    figure(1)
    ion()
    for index, T in enumerate([1, 5, 10, 50, 100, 200]):
        adaboost = aba.AdaBoost(DecisionStump, T)
        adaboost.train(X_train, y_train)
        subplot(2, 3, index + 1)
        decision_boundaries(adaboost, X_train, y_train, "Iteration: " + str(T))

    pause(8)

    best_iteration = val_error.index(np.min(val_error)) * 5
    print(best_iteration)
    ab = aba.AdaBoost(DecisionStump, best_iteration)
    ab.train(X_train, y_train)
    print(ab.error(X_test, y_test))
    return

def Q4(): # decision trees
    val_error = []
    train_error = []
    D = [3, 6, 8, 10, 12]

    return

def Q5(): # spam data
    # TODO - implement this function
    return

def main():
    Q3()
    Q4()
    Q5()

if __name__ == '__main__':
   main()


