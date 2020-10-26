import pandas as pd
import numpy as np
import cvxpy as cp

class MyClassifier:
    def __init__(self,K,M):
        self.K = K  #Number of classes
        self.M = M  #Number of features
        self.W = np.array([])
        self.w = np.array([])


    @staticmethod
    def corrupt_data(p,data):
        total_killed=0
        for i in range(data.shape[0]):
            row=data[i,:]
            number_to_kill=np.random.binomial(row.size,p)
            total_killed=total_killed+number_to_kill
            random_mask = np.random.choice(row.size, number_to_kill, replace=False)
            row[random_mask]=0
            data[i,:]=row
        return data, total_killed

        #     for j in range(data.shape[1]):
        #         kill=np.random.binomial(1, p)
        #         if(kill==0):
        #             data[i,j]=0
        #             num_deleted=num_deleted+1
        # return data, num_deleted


    @staticmethod
    def load_data(dataset_train,dataset_test):
        train_rawdata = np.genfromtxt(dataset_train, delimiter=",")
        test_rawdata = np.genfromtxt(dataset_test, delimiter=",") 
        train_data = train_rawdata[1:, 1:]
        train_label = train_rawdata[1:, 0]
        test_data = test_rawdata[1:, 1:]
        test_label = test_rawdata[1:, 0]
        return train_data, train_label, test_data, test_label


    #Used to turn a bigger dataset into a smaller one randomly
    @staticmethod
    def shrink_dataset(dataset,data_labels,portion):
        N = dataset.shape[0] #total number of elements we could use to train
        random_mask = np.random.choice(N, portion, replace=False)
        data_sample=dataset[random_mask,:]
        label_sample=data_labels[random_mask]
        return data_sample, label_sample

    def train_binaryClassification(self, the_label, train_data, train_label, verb=True):

        print("This is train_binaryClassification, you are training ", the_label, "...") #you can erase this line
        mask_train_the_label     = (train_label == the_label)
        mask_train_NOT_the_label = (train_label != the_label)

        new_train_label = np.array(train_label, copy=True)
        new_train_label[mask_train_the_label] = 1
        new_train_label[mask_train_NOT_the_label] = -1

        N = train_data.shape[0]
        
        t = cp.Variable(N)
        a = cp.Variable(self.M)
        b = cp.Variable(1)

        # formualte solve the opt problem
        obj = cp.Minimize(cp.sum(t))
        constraints = []

        for i in range(N):
            x_i = train_data[i,:]
            s_i = new_train_label[i]
            constraints.append(t[i] >= 1 - s_i * (x_i.T @ a + b))
            constraints.append(t[i] >= 1 - s_i * (x_i.T @ a + b))
            constraints.append(t[i] >= 0)

        prob2 = cp.Problem(obj, constraints)
        result = prob2.solve(verbose=verb)
        #print("\nThe optimal value is", prob2.value)
        return a.value, b.value

        #return W_value, w_value



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

        print("You are calling train()...") #you can erase this line

        label_list = np.unique(train_label)

        self.W = np.array([])
        self.w = np.array([])

        for the_label in label_list:
            a, b = self.train_binaryClassification(the_label, train_data, train_label,verb=False)
            self.W = np.append(self.W, a, axis=0)
            self.w = np.append(self.w, b, axis=0)
        print(self.W.shape)
        self.W = self.W.reshape((self.K,self.M))
        return

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

        f = np.argmax(self.W @ input.T  + self.w)

        return f

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

        N = test_data.shape[0]
        pred = np.zeros(N)

        for i in range(N):
            x_i = test_data[i,:]
            pred[i] = self.f(x_i)

        return pred


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

        print("You re calling TestCorrupted function") #you can erase this line

