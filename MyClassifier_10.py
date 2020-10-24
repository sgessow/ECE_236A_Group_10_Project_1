# Project 1 ECE 236A -*-
"""
ECE 236A Project 1, MyClassifier.py template. Note that you should change the
name of this file to MyClassifier_{groupno}.py
"""
import numpy as np
import cvxpy as cp
import itertools
import time


### Setup to Run Classifier Object

# CSV Importer
from numpy import genfromtxt

# Load Data from CSV
trainData = genfromtxt('/Users/sunaybhat/Dropbox/UCLA/Fall 2020/mnist_train.csv', delimiter=',')
testData = genfromtxt('mnist_test.csv', delimiter=',')

# Filter test data to 1 and 7
filteredData = trainData[np.logical_or(trainData[:,0] == 1 ,trainData[:,0] == 7),:]

train_data = filteredData[:,1:]
train_label= filteredData[:,0]

# Test 1 and 7 Case
testClass1 = MyClassifier(2,784)


class MyClassifier:
    def __init__(self,K,M):
        self.K = K  #Number of classes
        self.M = M  #Number of features
        self.W = []
        self.w = []
        
    def train(self, p, train_data, train_label):

    allClassLabels = np.unique(train_label)

    if np.size(allClassLabels) != self.K
        raise MyValidationError("Training Data Class Labels Does not match classes specified")

    for labels in list(itertools.combinations(allClassLabels,2)):

        # Number of datapoints
        n = train_data.shape[0]

        # Convert labels to -1,1 using sign function, y will be utilized in objective function
        y = np.sign(train_label - np.mean(labels))

        # Training Data as X
        x = train_data

        # Variables to optimimze (A vector of weights), b offset term
        A = cp.Variable(train_data.shape[1])
        b = cp.Variable()

        # Objective function to minimize
        objectiveF = cp.Minimize(cp.sum(1 + cp.multiply(y,(A @ x.T + b)))/testClass1.M)

        # Constraints: 1 + y * (Ax + b) >= 0 for every n
        constraints = []
        for iConstraint in range(1, n):
            constraints.append((1 - cp.multiply(y,(A @ x[iConstraint,:] - b))>= 0))


        prob = cp.Problem(objectiveFunction,constraints)

        start_time = time.time()
        result = prob.solve(verbose=True)
        tottime = time.time() - start_time
        self.W = A.value
        self.w = b.value
        print()
            
        
    def f(self,input):
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
        
        print()
        
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
        
        print()
    
    
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
        
        print()

    # function for displaying an image
    @staticmethod
    def display_number(input_vector, len_row):
        # pixels = letter_vector.reshape((len_row, -1))
        # plt.imshow(pixels, cmap='gray_r')
        # plt.show()
        # return 0
        pixels = (np.array(input_vector, dtype='float')).reshape(len_row, -1)
        img = Image.fromarray(np.uint8(pixels * -255), 'L')
        img = ImageOps.invert(img)
        img.show()
