# Project 1 ECE 236A -*-
"""
ECE 236A Project 1, MyClassifier.py template. Note that you should change the
name of this file to MyClassifier_{groupno}.py
"""

from itertools import combinations
import time
import numpy as np
import cvxpy as cp


class MyClassifier:
    def __init__(self,K,M):
        self.K = K  #Number of classes
        self.M = M  #Number of features
        self.W = [] #Feature Weights
        self.w = [] #OffsetValue
        self.ClassifierMap = list() #Label for each classifier, list of two tuples

    def train(self, p, train_data, train_label):

        start_time = time.time() # Delete Later

        # Determine number of unique labels in data set
        allClassLabels = np.unique(train_label)

        # Loop through unique pairs (Combinations) of labels to run one vs one training
        for iHyperPlane,(digit_i, digit_j) in enumerate(list(combinations(allClassLabels,2))):

            print('Runing Classifier: ', str(iHyperPlane + 1), 'for ', str((digit_i, digit_j))) # Delete Later

            # make mask labels for this iteration
            digit_mask = (train_label == digit_i) | (train_label == digit_j)

            # Training Data as X for digits
            X = train_data[digit_mask]

            # Convert labels to -1,1 using sign function and mean of labels, y will be utilized in constraints
            y = np.sign(train_label[digit_mask] - np.mean((digit_i, digit_j)))

            # Number of datapoints after filtering
            N = X .shape[0]

            # Number of features, Should be same as self.M
            M = X.shape[1]

            # Points to optimimze hyperplane (A vector of weights), b offset term
            A = cp.Variable(M)
            b = cp.Variable()

            # Substitution variable
            t = cp.Variable(N)

            # Objective function to minimize: sum of all vectors from classifier hyperplane
            objectiveF = cp.Minimize(cp.sum(t))

            # Multiple y through X to apply sign to every feature of every digit in X
            mat_Xsign = y[:, np.newaxis] * X

            # Constraints: t >=1 + y * (Ax + b) and t >= 0 for every digit in matrix form
            constraints = [t >= np.ones(N) + mat_Xsign @ A + y.T * b,
                           t >= np.zeros(N)]

            # Instantiate 'Problem' Class in CVXPY
            prob = cp.Problem(objectiveF, constraints)

            # Solve problem
            prob.solve(verbose=False)

            # Update Classifier attributes
            self.W.append(A.value)
            self.w.append(b.value)
            self.ClassifierMap.append((int(digit_i), int(digit_j)))

        # Shape weight arrays and return
        self.W = np.array(self.W)
        self.w = np.array(self.w)
        tottime = time.time() - start_time

        print('Done  training, total time =',round(tottime,2) , ' seconds')
            
    # Takes a Scalar input and hyperplane and outputs the class digit
    def f(self,input,vote_weight):

        scores = [0] * 10
        for i, val in enumerate(input):
            if vote_weight == False:
                if val >= 0:
                    scores[self.ClassifierMap[i][0]] += 1
                else:
                    scores[self.ClassifierMap[i][1]] += 1

            if vote_weight == True:
                if val >= 0:
                    scores[self.ClassifierMap[i][0]] += abs(val)
                else:
                    scores[self.ClassifierMap[i][1]] += abs(val)


        return np.argmax(scores)


    # Function to classify test data
    def classify(self,test_data,vote_weight=False):

        # Check for weights being trained
        if len(self.W) == 0 | len(self.w) == 0:
            print('Error: Weight (W and w) have not yet been trained in classifier!...')
            return -1

        # Initialize results array
        test_results = np.zeros(test_data.shape[0])

        # Loop through rows and pass in to f class function
        for iRow, test in enumerate(test_data):

            # Call f and store result
            test_results[iRow] = self.f(self.W @ test.flatten() + self.w,vote_weight)

        return test_results
    
    
    def TestCorrupted(self,p,test_data):

        print()