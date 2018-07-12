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
import bagging

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
    sample = [3, 6, 8, 10, 12]
    for samp in sample:
        dt = dta.DecisionTree(samp)
        dt.train(X_train, y_train)
        train_error.append(dt.error(X_train, y_train))
        val_error.append(dt.error(X_val, y_val))
    plot(sample, train_error)
    plot(sample, val_error)
    xlabel("samp")
    ylabel("error rate")
    legend(["train error", "validation error"], loc=5)
    show()
    # figure(1)
    # ion()
    for index, samp in enumerate(sample):
        dt = dta.DecisionTree(samp)
        dt.train(X_train, y_train)
        subplot(2,3, index + 1)
        decision_boundaries(dt, X_train, y_train, "samp = " + str(samp))
    pause(8)
    best_d = sample[val_error.index(np.min(val_error))]
    print(best_d)
    dt = dta.DecisionTree(best_d)
    dt.train(X_train, y_train)
    print(dt.error(X_test, y_test))
    # Bagging:
    val_error = []
    for B in range(5, 105, 5):
        print("B: " + str(B))
        bag = bagging.Bagging(dta.DecisionTree, B, best_d)
        bag.train(X_train, y_train)
        val_error.append(bag.error(X_val, y_val))

    plot(range(5, 105, 5), val_error)
    xlabel("B")
    ylabel("validation error rate")
    show()
    best_b = list(range(5, 105, 5))[val_error.index(np.min(val_error)) + 5]
    print("best b: ", best_b)
    bag = bagging.Bagging(dta.DecisionTree, best_b, best_d)
    bag.train(X_train, y_train)
    print(bag.error(X_test, y_test))


def Q5(): # spam data
    n_folds = 5
    # creating the data
    y = np.loadtxt("SpamData/spam.data", usecols=(-1,))
    y[y == 0] = -1
    data = np.loadtxt("SpamData/spam.data")
    X = data[:, :-1]
    sample_data_idx = np.random.permutation(np.array(range(len(X))))
    X_test, y_test, X_train, y_train = X[sample_data_idx[:1536]], y[sample_data_idx[:1536]],\
                                       X[sample_data_idx[1536:]], y[sample_data_idx[1536:]]
    T_values = [5, 50, 100, 200, 500, 1000]
    d_values = [5, 8, 10, 12, 15, 18]

    # splitting to folds
    cross_val_idx = np.random.permutation(np.array(range(len(X_train))))
    folds = np.array([X_train[cross_val_idx[int(i * len(X_train) / n_folds): int((i + 1) * len(X_train) / n_folds)]]
                      for i in range(n_folds)])
    folds_y = np.array([y_train[cross_val_idx[int(i * len(X_train) / n_folds): int((i + 1) * len(X_train) / n_folds)]]
                      for i in range(n_folds)])
    AB_mean_errors = []
    DT_mean_errors = []
    AB_sd = []
    DT_sd = []
    # cross validation:
    for val in range(len(T_values)):
        print("val" + str(val))
        # the len of T and d is the same
        ab_errors = []
        dt_errors = []
        for fold_idx in range(n_folds):
            print("fold idx " + str(fold_idx))
            fold_test = folds[fold_idx]
            fold_test_y = folds_y[fold_idx]
            other_fold_idx = [i for i in range(n_folds) if i!=fold_idx]
            other_folds = np.concatenate(folds[other_fold_idx])
            other_folds_y = np.concatenate(folds_y[other_fold_idx])

            # adaboost training on current fold:
            ab = aba.AdaBoost(DecisionStump, T_values[val])
            ab.train(other_folds, other_folds_y)
            ab_errors.append(ab.error(fold_test, fold_test_y))

            # DT training on current fold:
            dt = dta.DecisionTree(d_values[val])
            dt.train(other_folds, other_folds_y)
            dt_errors.append(dt.error(fold_test, fold_test_y))

        # add the mean errors:
        AB_mean_errors.append(np.array(ab_errors).mean())
        AB_sd.append(np.std(ab_errors))
        DT_mean_errors.append(np.array(dt_errors).mean())
        DT_sd.append(np.std(dt_errors))

    # plotting the errors in order to view the best T and d parameters:
    errorbar(T_values, AB_mean_errors, np.reshape(np.array(AB_sd), (len(T_values), 1)), ecolor="green")
    xlabel("parameter - T")
    ylabel("error rate")
    show()
    errorbar(d_values, DT_mean_errors, np.reshape(np.array(DT_sd), (len(T_values), 1)), ecolor="green")
    xlabel("parameter max depth")
    ylabel("error rate")
    show()
    # best_b = AB_mean_errors.index(np.min(AB_mean_errors))
    best_b = 200
    print(" best parameter for adaboost:" + str(best_b))
    ab = aba.AdaBoost(DecisionStump, best_b)
    ab.train(X_train, y_train)
    print(ab.error(X_test, y_test))
    # best_b = DT_mean_errors.index(np.min(DT_mean_errors))
    best_b = 10
    print("DT best parameter:" + str(best_b))
    dt = dta.DecisionTree(best_b)
    dt.train(X_train, y_train)
    print(dt.error(X_test, y_test))

def main():
    # Q3()
    # Q4()
    Q5()

if __name__ == '__main__':
   main()


