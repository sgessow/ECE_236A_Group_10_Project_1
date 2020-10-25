from MyClassifier_10 import MyClassifier
import numpy as np
import cvxpy as cp
import itertools
import time

# CSV Importer
from numpy import genfromtxt
# Image Viewer
from PIL import Image, ImageOps

# Load Data from CSV
trainData = genfromtxt('/Users/sunaybhat/Dropbox/UCLA/Fall 2020/mnist_train.csv', delimiter=',')
testData = genfromtxt('mnist_test.csv', delimiter=',')

# Get Rid of Headers
trainData = trainData[1:]
testData = testData[1:]


### 1,7 and 8
digit_mask = (trainData[:,0] == 1) | (trainData[:,0] == 7) | (trainData[:,0] == 8)
filteredData = trainData[digit_mask,:]
train_data = filteredData[:,1:]
train_label = filteredData[:,0]

Class1 = MyClassifier(3,784)
Class1.train(0.2,train_data,train_label)

# Filter test data to 1, 7, and 8
digit_mask = (testData[:,0] == 1) | (testData[:,0] == 7) | (testData[:,0] == 8)
test_data = testData[digit_mask,:]
test_labels = test_data[:,0]
test_labels = test_labels.reshape(3137,1)
results = Class1.classify(test_data)