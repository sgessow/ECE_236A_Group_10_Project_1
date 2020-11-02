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

    def train(self, p, train_data, train_label,ntrains =1):

        start_time = time.time() # Delete Later

        # Determine number of unique labels in data set
        allClassLabels = np.unique(train_label)

        # Loop through unique pairs (Combinations) of labels to run one vs one training
        for iHyperPlane,(digit_i, digit_j) in enumerate(list(combinations(allClassLabels,2))):

            print('Runing Classifier: ', str(iHyperPlane + 1), 'for ', str((digit_i, digit_j)))  # Delete Later

            # Setup variables to store sets of averaging
            tot_W = []
            tot_w = []

            # Run through number of trainings to average later
            for i in range(0,ntrains):

                # make mask labels for this iteration
                digit_mask = (train_label == digit_i) | (train_label == digit_j)

                # Training Data as X for digits
                X = train_data[digit_mask]

                # Corrupt Data
                if p > 0:
                    X = self.TrainCorrupted(0.6,X)
                    print('Training with corrupted data ')

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

                tot_W.append(A.value)
                tot_w.append(b.value)

            # Update Classifier attributes
            self.W.append(np.mean(tot_W,axis=0))
            self.w.append(np.mean(tot_w,axis=0))
            self.ClassifierMap.append((int(digit_i), int(digit_j)))

        # Shape weight arrays and return
        self.W = np.array(self.W)
        self.w = np.array(self.w)
        tottime = time.time() - start_time

        print('Done  training, total time =',round(tottime,2) , ' seconds')
            
    # Takes a Scalar input and hyperplane and outputs the class digit
    def f(self,input):

        # Vector to tabulate classification
        votes  = [0] * 10

        for i, val in enumerate(input):
            if val >= 0:
                votes[self.ClassifierMap[i][0]] += 1

            else:
                votes[self.ClassifierMap[i][1]] += 1

        # Return index/digit with most classificatons/votes
        return np.argmax(votes)


    # Function to classify test data
    def classify(self,test_data):

        # Check for weights being trained
        if len(self.W) == 0 | len(self.w) == 0:
            print('Error: Weight (W and w) have not yet been trained in classifier!...')
            return -1

        # Initialize results array
        test_results = np.zeros(test_data.shape[0])

        # Loop through rows and pass in to f class function
        for iRow, test in enumerate(test_data):

            # Call f and store result
            test_results[iRow] = self.f(self.W @ test.flatten() + self.w)

        return test_results
    
    
    def TestCorrupted(self,p,test_data):

        # Setup random matrix to corrupt data
        random_mask = np.random.random(test_data.shape) > p
        total_killed = np.sum(random_mask)

        # Corrupt data
        corrupt_Data = test_data * random_mask

        print('Killed ', round((1 - total_killed/(corrupt_Data.shape[0]*784)) * 100,2), '% Pixels of test data...')

        return self.classify(corrupt_Data)

    def TrainCorrupted(self,p,train_data,n_sets=1):

        # Setup random matrix to corrupt data and loop through sets desired
        corrupt_Data = np.empty((0,784))
        total_killed = 0
        for i in range(0,n_sets):
            random_mask = np.random.random(train_data.shape) > p
            total_killed += np.sum(random_mask)

            # Corrupt data and return
            corrupt_Data = np.append(corrupt_Data, train_data * random_mask,axis=0)

        print('Killed ', round((1 - total_killed / (corrupt_Data.shape[0] * 784)) * 100, 2), '% Pixels of training data...')

        return corrupt_Data