# Project 1 ECE 236A -*-
"""
ECE 236A Project 1, MyClassifier.py template. Note that you should change the
name of this file to MyClassifier_{10}.py
"""

import numpy as np
import cvxpy as cp
from itertools import combinations # Added for classifier combinations


class MyClassifier:
    def __init__(self,K,M):
        self.K = K  #Number of classes
        self.M = M  #Number of features
        self.W = [] #Feature Weights
        self.w = [] #OffsetValue
        self.ClassifierMap = list() #Label for each classifier, list of two tuples

    # Training function
    def train(self, p, train_data, train_label):


        # Determine number of unique labels in data set
        allClassLabels = np.unique(train_label)

        # Loop through unique pairs (Combinations) of labels to run one vs one training
        for iHyperPlane,(digit_i, digit_j) in enumerate(list(combinations(allClassLabels,2))):

            print('Runing Classifier: ', str(iHyperPlane + 1), 'for ', str((digit_i, digit_j)))  # Delete Later

            # make mask labels for this iteration
            digit_mask = (train_label == digit_i) | (train_label == digit_j)

            # Training Data as X for digits
            X = train_data[digit_mask]

            # Corrupt data and train based on value of p
            if p < 0.2:
                ntile = 1
                X,_ = self.apply_error(p,X,1)
            else:
                ntile = 3
                X, _ = self.apply_error(p - 0.2, X, 3)

            # Convert labels to -1,1 using sign function and mean of labels
            y = np.sign(train_label[digit_mask] - np.mean((digit_i, digit_j)))
            y = np.tile(y, ntile)

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

            tol_goal = 1e-1
            tol_ok = .5
            prob.solve(solver="ECOS", max_iters=75, abstol=tol_goal, reltol=tol_goal, feastol=tol_goal,
                                abstol_inacc=tol_ok, reltol_inacc=tol_ok, feastol_inacc=tol_ok * 2, verbose=False)


            # Update Classifier attributes
            self.W.append(A.value)
            self.w.append(b.value)
            self.ClassifierMap.append((int(digit_i), int(digit_j)))

        # Shape weight arrays and return
        self.W = np.array(self.W)
        self.w = np.array(self.w)

        print('Done  training, yay!...')

    # Takes a Scalar input and hyperplane and outputs the class digit
    def f(self,input):

        # Vector to tabulate classification
        votes  = [0] * 10

        # Loop trough each classifier raw value result and assign a vote to digit index
        for i, val in enumerate(input):
            if val >= 0:
                votes[self.ClassifierMap[i][0]] += 1

            else:
                votes[self.ClassifierMap[i][1]] += 1

        # Return index/digit with most classificatons/votes
        return np.argmax(votes)

    # Function to classify test data
    def classify(self,test_data):

        # Remove labels if extra columns found
        while test_data.shape[1] > self.M:
            test_data = test_data[:, 1:]

        # Check for weights being trained
        if len(self.W) == 0 | len(self.w) == 0:
            print('Error: Weight (W and w) have not yet been trained in classifier!...')
            return -1

        # Initialize results array
        test_results = np.zeros(test_data.shape[0])

        # Loop through rows and pass into f function for result
        for iRow, test in enumerate(test_data):

            # Call f and store result
            test_results[iRow] = self.f(self.W @ test.flatten() + self.w)

        return test_results

    # Function to corrupt test data
    def TestCorrupted(self,p,test_data):

        # Remove labels if extra columns find
        while test_data.shape[1] > self.M:
            test_data = test_data[:, 1:]

        # Pass test data into apply error
        corrupt_Data, total_killed = self.apply_error(p, test_data)

        return self.classify(corrupt_Data)

    # A function to corrupt data, default removes ~(p*100)% of pixels, can construct larger training data set
    #   based on appending corrupted matrices together
    @staticmethod
    def apply_error(p, images,k=1):
        image_sets = [] # returned array
        feature_erased_list = [] # Features corrupted, used for histogram

        # Loop through copies to be made, default 1 for testing/basic corruption
        for i in range(k):

            # Generate corruption mask for pixels grater than p in random matrix
            corrupt_mask = np.random.random(images.shape) >= p

            # Count features erased per image
            feature_erased_list = images.shape[1] * np.ones(images.shape[0]) - np.sum(corrupt_mask, axis=1)

            # Multiply corruption mask
            corrupt_images = images * corrupt_mask

            # Append corrupted image to return array
            image_sets.append(corrupt_images)

            # Iterate through p's for training purposed
            p += 0.2

        return np.concatenate(image_sets, axis=0), feature_erased_list