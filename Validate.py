## Validation functions script
from numpy import genfromtxt
from PIL import Image, ImageOps
import MyClassifier_10 as MyClassifier
import numpy as np
from importlib import reload



testData = genfromtxt('mnist_test.csv', delimiter=',')
# Get Rid of Headers
testData = testData[1:]

# Load Data from CSV, change path to train csv
trainData = genfromtxt('/Users/sunaybhat/Dropbox/UCLA/Fall 2020/mnist_train.csv', delimiter=',')
# Get Rid of Headers
trainData = trainData[1:]

def CorruptTrain_1and7(ntrains):
    reload(MyClassifier)
    accuracy = list()
    # Validate 1 and 7
    Class2_Corr_1and7 = MyClassifier.MyClassifier(2,784)
    digit_mask = (trainData[:, 0] == 1) | (trainData[:, 0] == 7)  # Filter Digits Down
    filteredData = trainData[digit_mask, :]
    train_data = filteredData[:, 1:]
    train_label = filteredData[:, 0]
    Class2_Corr_1and7.train(0.6, train_data, train_label)

    digit_mask = (testData[:, 0] == 1) | (testData[:, 0] == 7)  # Filter Digits Down
    filteredTestData = testData[digit_mask, :]
    test_data = filteredTestData[:, 1:]
    test_labels = filteredTestData[:, 0]
    results = Class2_Corr_1and7.classify(test_data)
    correct = results == test_labels
    accuracy.append(np.sum(correct) / len(test_labels) * 100)
    # print('Class2_Corr_1and7 Success for 1 and 7 is {}%...'.format(round(accuracy, 1)))

    results = Class2_Corr_1and7.TestCorrupted(0.4,test_data)
    correct = results == test_labels
    accuracy.append(np.sum(correct) / len(test_labels) * 100)
    # print('Class2_Corr_1and7 Success for p = 0.2 is {}%...'.format(round(accuracy, 1)))

    results = Class2_Corr_1and7.TestCorrupted(0.6,test_data)
    correct = results == test_labels
    accuracy.append(np.sum(correct) / len(test_labels) * 100)
    # print('Class2_Corr_1and7 Success for p = 0.4 is {}%...'.format(round(accuracy, 1)))

    results = Class2_Corr_1and7.TestCorrupted(0.8,test_data)
    correct = results == test_labels
    accuracy.append(np.sum(correct) / len(test_labels) * 100)
    # print('Class2_Corr_1and7 Success for p = 0.6 is {}%...'.format(round(accuracy, 1)))

    return accuracy

def compareClass10_train(p):
    reload(MyClassifier)
    train_data = trainData[:, 1:]
    train_label = trainData[:, 0]
    Class10 = MyClassifier.MyClassifier(10, 784)
    Class10.train(0.6, train_data, train_label)

    # Test
    accuracy = list()
    test_data = testData[:, 1:]
    test_labels = testData[:, 0]
    results = Class10.classify(test_data)
    correct = results == test_labels
    accuracy.append(np.sum(correct) / len(test_labels) * 100)

    results = Class10.TestCorrupted(0.4,test_data)
    correct = results == test_labels
    accuracy.append(np.sum(correct) / len(test_labels) * 100)

    results = Class10.TestCorrupted(0.6,test_data)
    correct = results == test_labels
    accuracy.append(np.sum(correct) / len(test_labels) * 100)

    results = Class10.TestCorrupted(0.8,test_data)
    correct = results == test_labels
    accuracy.append(np.sum(correct) / len(test_labels) * 100)

    return accuracy

