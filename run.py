from MyClassifier_10 import MyClassifier

# CSV Importer
from numpy import genfromtxt

import dill
dill.dump_session('TrainedClassifier.pkl')


### Setup ##########
# Load Data from CSV
trainData = genfromtxt('/Users/sunaybhat/Dropbox/UCLA/Fall 2020/mnist_train.csv', delimiter=',')
testData = genfromtxt('mnist_test.csv', delimiter=',')

# Get Rid of Headers
trainData = trainData[1:]
testData = testData[1:]

#############

### All Digits #############
filteredData = trainData[:]
train_data = filteredData[:,1:]
train_label = filteredData[:,0]
Class10 = MyClassifier(10,784)
Class10.train(0.2,train_data,train_label)

# Test
test_data = testData[:,1:]
test_labels = testData[:,0]
test_labels = test_labels.reshape(test_labels.shape[0],1)
results = Class10.classify(test_data)

accuracy = np.sum(abs(results == test_labels.T))/len(test_labels) * 100
print('Test Data Success 0-9 is {}%...'.format(round(accuracy,1)))

### 1 and 7 #########
digit_mask = (trainData[:,0] == 1) | (trainData[:,0] == 7) # Filter Digits Down
filteredData = trainData[digit_mask,:]
train_data = filteredData[:,1:]
train_label = filteredData[:,0]
Class1and7 = MyClassifier(2,784)
Class1and7.train(0.2,train_data,train_label)

# Test
digit_mask = (testData[:,0] == 1) | (testData[:,0] == 7) # Filter Digits Down
filteredTestData = testData[digit_mask, :]
test_data = filteredTestData[:,1:]
test_labels = filteredTestData[:,0]
test_labels = test_labels.reshape(test_labels.shape[0],1)
results = Class1and7.classify(test_data)

accuracy = np.sum(abs(results == test_labels))/len(test_labels) * 100

print('Test Data Success is {}%...'.format(round(accuracy,1)))

### 1,7 and 8
digit_mask = (trainData[:,0] == 1) | (trainData[:,0] == 7) | (trainData[:,0] == 8)
filteredData = trainData[digit_mask,:]
train_data = filteredData[:,1:]
train_label = filteredData[:,0]

Class3 = MyClassifier(3,784)
Class3.train(0.2,train_data,train_label)

# Filter test data to 1, 7, and 8
digit_mask = (testData[:,0] == 1) | (testData[:,0] == 7) | (testData[:,0] == 8)
filteredTestData = testData[digit_mask,:]
test_data = filteredTestData[:,1:]
test_labels = filteredTestData[:,0]
results = Class3.classify(test_data)
accuracy = np.sum(abs(results == test_labels))/len(test_labels) * 100
print('Test Data Success is {}%...'.format(round(accuracy,1)))

### 1,7,8, and 9
digit_mask = (trainData[:,0] == 1) | (trainData[:,0] == 7) | (trainData[:,0] == 8) | (trainData[:,0] == 9)
filteredData = trainData[digit_mask,:]
train_data = filteredData[:,1:]
train_label = filteredData[:,0]

Class4 = MyClassifier(4,784)
Class4.train(0.2,train_data,train_label)

# Filter test data to 1, 7, and 8
digit_mask = (testData[:,0] == 1) | (testData[:,0] == 7) | (testData[:,0] == 8) | (testData[:,0] == 9)
filteredTestData = testData[digit_mask,:]
test_data = filteredTestData[:,1:]
test_labels = filteredTestData[:,0]
results = Class4.classify(test_data)
accuracy = np.sum(abs(results == test_labels))/len(test_labels) * 100
print('Test Data 4 class Success is {}%...'.format(round(accuracy,1)))

### All but 9, 9 classes
digit_mask = (trainData[:,0] != 9)
filteredData = trainData[digit_mask,:]
train_data = filteredData[:,1:]
train_label = filteredData[:,0]
Class9 = MyClassifier(9,784)
Class9.train(0.2,train_data,train_label)

# Test
digit_mask = (testData[:,0] != 9)
filteredTestData = testData[digit_mask,:]
test_data = filteredTestData[:,1:]
test_labels = filteredTestData[:,0]
results = Class10.classify(test_data)
accuracy = np.sum(abs(results == test_labels.T))/len(test_labels) * 100
print('Test Data 4 class Success is {}%...'.format(round(accuracy,1)))

# # function for displaying an image
# @staticmethod
# def display_number(input_vector, len_row):
#     # pixels = letter_vector.reshape((len_row, -1))
#     # plt.imshow(pixels, cmap='gray_r')
#     # plt.show()
#     # return 0
#     pixels = (np.array(input_vector, dtype='float')).reshape(len_row, -1)
#     img = Image.fromarray(np.uint8(pixels * -255), 'L')
#     img = ImageOps.invert(img)
#     img.show()