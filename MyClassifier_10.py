# Project 1 ECE 236A -*-
"""
ECE 236A Project 1, MyClassifier.py template. Note that you should change the
name of this file to MyClassifier_{groupno}.py
"""


class MyClassifier:
    def __init__(self,K,M):
        self.K = K  #Number of classes
        self.M = M  #Number of features
        self.W = []
        self.w = []
        self.OnevOneLabel = list()

    def train(self, p, train_data, train_label, time, itertools, np, cp):

        start_time = time.time()

        # Determine number of unique labels in data set
        allClassLabels = np.unique(train_label)

        # Loop through unique paris (Combinations) of labels to run one vs one training
        for iHyper,labels in enumerate(list(itertools.combinations(allClassLabels,2))):

            print('Runing Classifier: ', str(iHyper + 1)) # Delete Later

            # make mask labels for this iteration
            mask = (train_label == labels[0]) | (train_label == labels[1])

            # Convert labels to -1,1 using sign function and mean of labels, y will be utilized in constraints
            y = np.sign(train_label[mask] - np.mean(labels))

            # Training Data as X
            X = train_data[mask,:]

            # Number of datapoints after filtering
            n = X .shape[0]

            # Points to optimimze hyperplane (A vector of weights), b offset term
            A = cp.Variable(X.shape[1])
            b = cp.Variable()

            # Substitution variable
            t = cp.Variable(n)

            # Objective function to minimize: sum of all vectors from classifier hyperplane
            objectiveF = cp.Minimize(cp.sum(t))

            # Constraints: t >=1 + y * (Ax + b) and t >= 0 for every n
            constraints = []
            for i, (digit, label) in enumerate(zip(X, y)):
                flat_digit = digit.flatten()
                constraints.append(t[i] >= (1 + y[i] * (flat_digit.T @ A + b)))
                constraints.append(t[i] >= 0)

            # Instantiate 'Problem' Class in CVXPY
            prob = cp.Problem(objectiveF, constraints)

            # Solve problem
            prob.solve(verbose=False)

            # Update Classifier attributes

            self.W = np.append(self.W,A.value)
            self.w = np.append(self.w,b.value)
            self.OnevOneLabel.append(labels)

        self.W = self.W.reshape(self.M,-1)
        tottime = time.time() - start_time
        print('Done  training, total time =',round(tottime,2) , ' seconds')
            
    # Takes a Scalar input and hyperplane and outputs the class digit
    def f(self,input,iHyper):

        if input >= 0:
            return self.OnevOneLabel[iHyper][0]
        else:
            return self.OnevOneLabel[iHyper][1]

    # Function to classify test data
    def classify(self,test_data,time,np):

        start_time = time.time()

        # Remove Labels
        test_data = test_data[:, 1:]

        # Initialize results array
        test_results = np.zeros([test_data.shape[0], len(self.OnevOneLabel)])

        # Loop through hyperplanes, One v One
        for iHyper, labels in enumerate(self.OnevOneLabel):

            # Loop through rows, f() takes only one scalar input
            for iRow in range(0, test_data.shape[0]):

                # Fill out results array for every row and for every hyperplane
                test_results[iRow, iHyper] = self.f(test_data[iRow, :] @ self.W[:, iHyper] + self.w[iHyper], iHyper)

        # One ve One most common results implementation (vote by class), take mode of matrix in hyperplane dimension
        test_results, _ = self.mode(test_results,np, axis=1)

        tottime = time.time() - start_time
        print('Done Classification,  total time =',round(tottime,2) , ' seconds')

        return test_results
    
    
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

    @staticmethod
    def load_filter_data(train_data, train_label):
        my_data = genfromtxt(train_data, delimiter=',')
        np.delete(my_data, 0, axis=0)
        lables = my_data[:, 0]
        good_rows = np.where(lables == train_label[0])
        for i in train_label[1::]:
            good_rows = np.append(good_rows, np.where(lables == i))
        good_rows.sort()
        my_data = np.take(my_data, good_rows, axis=0)
        return my_data

    @staticmethod
    def mode(a, np, axis=0):
        scores = np.unique(np.ravel(a))
        testshape = list(a.shape)
        testshape[axis] = 1
        oldmostfreq = np.zeros(testshape)
        oldcounts = np.zeros(testshape)

        for score in scores:
            template = (a == score)
            counts = np.expand_dims(np.sum(template, axis),axis)
            mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
            oldcounts = np.maximum(counts, oldcounts)
            oldmostfreq = mostfrequent

        return mostfrequent, oldcounts