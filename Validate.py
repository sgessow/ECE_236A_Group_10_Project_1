## Validation functions script
from numpy import genfromtxt
from PIL import Image, ImageOps
from MyClassifier_10 import MyClassifier
import numpy as np
testData = genfromtxt('mnist_test.csv', delimiter=',')
# Get Rid of Headers
testData = testData[1:]

def display_number(input_vector, len_row):
    # pixels = letter_vector.reshape((len_row, -1))
    # plt.imshow(pixels, cmap='gray_r')
    # plt.show()
    # return 0
    pixels = (np.array(input_vector, dtype='float')).reshape(len_row, -1)
    img = Image.fromarray(np.uint8(pixels * -255), 'L')
    img = ImageOps.invert(img)
    img.show()

def testClass10_full(Class10,show_fails = False):
    # Test all
    test_data = testData[:, 1:]
    test_labels = testData[:, 0]
    results = Class10.classify(test_data)
    correct = results == test_labels
    accuracy = np.sum(correct) / len(test_labels) * 100
    print('Class10 Success for all digits is {}%...'.format(round(accuracy, 1)))

    # Shoe three failing digits
    if show_fails:
        locFails = np.array(np.where(correct.T == 0))[:, 0:3]
        display_number(test_data[locFails,:], 28*3)
        print('First 3 Incorrectly Classified as: ', str(results[locFails]))

    # Test 1 and 7
    digit_mask = (testData[:, 0] == 1) | (testData[:, 0] == 7)  # Filter Digits Down
    filteredTestData = testData[digit_mask, :]
    test_data = filteredTestData[:, 1:]
    test_labels = filteredTestData[:, 0]
    results = Class10.classify(test_data)
    correct = results == test_labels
    accuracy = np.sum(correct) / len(test_labels) * 100
    print('Class10 Success for 1 and 7 is {}%...'.format(round(accuracy, 1)))

    if show_fails:
        locFails = np.array(np.where(correct.T == 0))[:, 0:3]
        display_number(test_data[locFails, :], 28 * 3)
        print('First 3 Incorrectly Classified as: ', str(results[locFails]))

    # Test Random Combo
    K = int(np.floor((np.random.random(1) * 10)[0])+1)  # Number of Classes
    Digits = list(np.floor((np.random.random(K) * 10)[0:K]))
    print('Randomly choosing ', K, ' # of classes and', Digits, ' digits to Test')
    stringMask = ''
    for i in Digits:
        stringMask = stringMask + '(testData[:, 0] ==' + str(int(i)) + ') | '
    digitMask = eval(stringMask[:-2])
    filteredTestData = testData[digit_mask, :]
    test_data = filteredTestData[:, 1:]
    test_labels = filteredTestData[:, 0]
    results = Class10.classify(test_data)
    correct = results == test_labels
    accuracy = np.sum(correct) / len(test_labels) * 100
    print('Class10 Success random is {}%...'.format(round(accuracy, 1)))

    if show_fails:
        locFails = np.array(np.where(correct.T == 0))[:, 0:3]
        display_number(test_data[locFails, :], 28 * 3)
        print('First 3 Incorrectly Classified as: ', str(results[locFails]))

def validate_1and7(show_fails = False):
    # Validate 1 and 7
    Class2_1and7 = MyClassifier(2,748)
    # Load Data from CSV, change path to train csv
    trainData = genfromtxt('/Users/sunaybhat/Dropbox/UCLA/Fall 2020/mnist_train.csv', delimiter=',')
    # Get Rid of Headers
    trainData = trainData[1:]

    digit_mask = (trainData[:, 0] == 1) | (trainData[:, 0] == 7)  # Filter Digits Down
    filteredData = trainData[digit_mask, :]
    train_data = filteredData[:, 1:]
    train_label = filteredData[:, 0]
    Class2_1and7.train(0.2, train_data, train_label)


    digit_mask = (testData[:, 0] == 1) | (testData[:, 0] == 7)  # Filter Digits Down
    filteredTestData = testData[digit_mask, :]
    test_data = filteredTestData[:, 1:]
    test_labels = filteredTestData[:, 0]
    results = Class2_1and7.classify(test_data)
    correct = results == test_labels
    accuracy = np.sum(correct) / len(test_labels) * 100
    print('Class10 Success for 1 and 7 is {}%...'.format(round(accuracy, 1)))

    if show_fails:
        locFails = np.array(np.where(correct.T == 0))[:, 0:3]
        display_number(test_data[locFails, :], 28 * 3)
        print('First 3 Incorrectly Classified as: ', str(results[locFails]))
