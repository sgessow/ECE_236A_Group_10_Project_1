# Project 1 ECE 236A -*-
"""
ECE 236A Project 1, MyClassifier.py template. Note that you should change the
name of this file to MyClassifier_{groupno}.py
"""
from itertools import combinations
import time

import pandas as pd
import numpy as np
import cvxpy as cp

rep = 1

class MyClassifier:
    def __init__(self, K, M):
        self.K = K  #Number of classes
        self.M = M  #Number of features
        self.W = []
        self.w = []
        self.pair_mapping = []
        
    def train(self, p, train_data, train_label):
        
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

        for i, (digit_i, digit_j) in enumerate(combinations(range(self.K), 2)):
            digit_mask = (train_label == digit_i) | (train_label == digit_j)
            digit_subset = train_data[digit_mask]
            label_subset = train_label[digit_mask]
            #corrupt_subset = corrupt_images(images, p, rep)
            #corrupt_labels = np.tile(label_subset, (rep,))

            N = digit_subset.shape[0]

            input_dim = digit_subset.shape[1]

            t = cp.Variable(N)
            a = cp.Variable(input_dim)
            b = cp.Variable()

            mat_1 = np.zeros((N, input_dim))
            mat_2 = np.zeros(N)

            obj = cp.Minimize(cp.sum(t))
            constraints = []
            for i, (digit, label) in enumerate(zip(digit_subset, label_subset)):
                flat_digit = digit.flatten()
                if label == digit_i:
                    mat_1[i,:] = -flat_digit.T
                    mat_2[i] = -1
                else:
                    mat_1[i,:] = flat_digit.T
                    mat_2[i] = 1
            constraints = [t >= np.ones(N) + mat_1 @ a + mat_2.T * b,
                           t >= np.zeros(N)]
            start_time = time.time()
            prob = cp.Problem(obj, constraints)

            result = prob.solve(solver='SCS', verbose=True)
            tottime = time.time() - start_time
            print("Solve Time:", tottime)

            self.W.append(a.value)
            self.w.append(b.value)
            self.pair_mapping.append((digit_i, digit_j))

        self.W = np.array(self.W)
        self.w = np.array(self.w)

        
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
        scores = [0] * 10
        for i, val in enumerate(input):
            if val >= 0:
                scores[self.pair_mapping[i][0]] += 1
            else:
                scores[self.pair_mapping[i][1]] += 1

        return np.argmax(scores)
        
        
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


# A function which applies noise to the training or testing data
# k specifies how many corrupted versions of each point to generate
def apply_error(images, p, k=1):
    image_sets = []
    for i in range(k):
        corrupt_images = images.copy()
        error = np.random.random(images.shape) >= p
        corrupt_images = corrupt_images*error
        image_sets.append(corrupt_images)
    
    return np.concatenate(image_sets, axis=0)
