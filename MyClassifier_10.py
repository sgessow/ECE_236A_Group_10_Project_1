# Project 1 ECE 236A -*-
"""
ECE 236A Project 1, MyClassifier.py template. Note that you should change the
name of this file to MyClassifier_{groupno}.py
"""
import pandas as pd
import numpy as np
import cvxpy as cp

#
from numpy import genfromtxt
#used for displaying the image
#from matplotlib import pyplot as plt
from PIL import Image, ImageOps


class MyClassifier:
    def __init__(self,K,M):
        self.K = K  #Number of classes
        self.M = M  #Number of features
        self.W = []
        self.w = []
    
    @staticmethod
    def load_filter_data(train_data, train_label):
        my_data = genfromtxt(train_data, delimiter=',')
        np.delete(my_data,0,axis=0)
        lables=my_data[:,0]
        good_rows=np.where(lables==train_label[0])
        for i in train_label[1::]:
            good_rows=np.append(good_rows,np.where(lables==i))
        good_rows.sort()
        my_data=np.take(my_data,good_rows,axis=0)
        return my_data
    
    #function for displaying an image
    @staticmethod
    def display_letter(input_vector, len_row):
        # pixels = letter_vector.reshape((len_row, -1))
        # plt.imshow(pixels, cmap='gray_r')
        # plt.show()
        # return 0
        pixels = (np.array(input_vector, dtype='float')).reshape(len_row,-1)
        img = Image.fromarray(np.uint8(pixels * -255) , 'L')
        img=ImageOps.invert(img)
        img.show()



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
        #you can erase this line

        # Load Data from CSV and only keep the good lines
        my_data=MyClassifier.load_filter_data(train_data, train_label)
        MyClassifier.display_letter(my_data[100][1::],28)   
        print(my_data[100][0])  
        
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
        
        print() #you can erase this line
        
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
        
        print() #you can erase this line
    
    
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




def main():
    C=MyClassifier(2,4)
    C.train(.6,"mnist_train.csv",[1,7])
    print("done")

if __name__ == "__main__":
    main()