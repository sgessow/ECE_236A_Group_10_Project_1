# Project 1 ECE 236A -*-
"""
ECE 236A Project 1, MyClassifier.py template. Note that you should change the
name of this file to MyClassifier_{groupno}.py
"""
import time

import pandas as pd
import numpy as np
import cvxpy as cp

class MyClassifier:
    def __init__(self,K,M):
        self.K = K  #Number of classes
        self.M = M  #Number of features
        self.W = []
        self.w = []
        
    def train(self, p, digit1, digit2, train_data, train_label):
        
        # THIS IS WHERE YOU SHOULD WRITE YOUR TRAINING FUNCTION
        #
        # The inputs to this function are:
        #
        # self: a reference to the classifier object.
        # train_data: a matrix of dimesions N_train x M, where N_train
        # is the number of inputs used for training. Each row is an
        # input vector.
        # trainLabel: a vector of length N_train. Each element is the
        # label for the corresponding input column vector in trainData.
        #
        # Make sure that your code sets the classifier parameters after
        # training. For example, your code should include a line that
        # looks like "self.W = a" and "self.w = b" for some variables "a"
        # and "b".
        digit_mask = (train_label == digit1) | (train_label == digit2)
        digit_subset = train_data[digit_mask]
        label_subset = train_label[digit_mask]

        self.digit1 = digit1
        self.digit2 = digit2
        N = digit_subset.shape[0]

        input_dim = digit_subset.shape[1] * digit_subset.shape[2]

        t = cp.Variable(N)
        a = cp.Variable(input_dim)
        b = cp.Variable()

        obj = cp.Minimize(cp.sum(t))
        constraints = []
        for i, (digit, label) in enumerate(zip(digit_subset, label_subset)):
            flat_digit = digit.flatten()
            if label == digit1:
                constraints.append(t[i] >= 1 - (flat_digit.T @ a + b))
            else:
                constraints.append(t[i] >= 1 + (flat_digit.T @ a + b))
            constraints.append(t[i] >= 0)
        start_time = time.time()
        prob = cp.Problem(obj, constraints)
        tottime = time.time() - start_time

        start_time = time.time()
        result = prob.solve(verbose=True)
        tottime = time.time() - start_time
        self.W = a.value
        self.w = b.value
        
    def f(self, input):
        # THIS IS WHERE YOU SHOULD WRITE YOUR CLASSIFICATION FUNCTION
        #
        # The inputs of this function are:
        #
        # input: the input to the function f(*), equal to g(y) = W^T y + w
        #
        # The outputs of this function are:
        #
        # s: this should be a scalar equal to the class estimated from
        # the corresponding input data point, equal to f(W^T y + w)
        # You should also check if the classifier is trained i.e. self.W and
        # self.w are nonempty
        if input >= 0:
            return self.digit1
        else:
            return self.digit2
        
        
    def classify(self,test_data):
        # THIS FUNCTION OUTPUTS ESTIMATED CLASSES FOR A DATA MATRIX
        # 
        # The inputs of this function are:
        # self: a reference to the classifier object.
        # test_data: a matrix of dimesions N_test x M, where N_test
        # is the number of inputs used for training. Each row is an
        # input vector.
        #
        #
        # The outputs of this function are:
        #
        # test_results: this should be a vector of length N_test,
        # containing the estimations of the classes of all the N_test
        # inputs.
        test_results = np.zeros(test_data.shape[0])
        for i, test in enumerate(test_data):
            test_results[i] = self.f(self.W @ test.flatten() + self.w)

        return test_results    
    
    def TestCorrupted(self,p,test_data):
        # THIS FUNCTION OUTPUTS ESTIMATED CLASSES FOR A DATA MATRIX
        #
        #
        # The inputs of this function are:
        #
        # self: a reference to the classifier object.
        # test_data: a matrix of dimesions N_test x M, where N_test
        # is the number of inputs used for training. Each row is an
        # input vector.
        #
        # p:erasure probability
        #
        #
        # The outputs of this function are:
        #
        # test_results: this should be a vector of length N_test,
        # containing the estimations of the classes of all the N_test
        # inputs.
        
        print() #you can erase this line
