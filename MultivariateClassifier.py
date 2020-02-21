import numpy as np
from sklearn.decomposition import PCA

'''
Multivariate Classifier for CS 412

This script implements a multivariate classifier based on parameter estimation.
The scrip classifies by using disciminant functions for each class, assigning
a given data matrix to the class most likely given the discriminant methods. 
Only one library is used, numpy.


@author Thomas Leuenberger and Rohit Saikrishnan Ramesh
'''

# Flag for pca mode
run_PCA = False

if run_PCA == True:
    #PCA section written by @Rohit Saikrishnan
    # Import Data
    pca = PCA(n_components = 50)
    training_labels = np.genfromtxt('train.csv', delimiter=",", usecols=-1, dtype='unicode', skip_header=1, autostrip=True)
    training_data = np.genfromtxt('train.csv', delimiter=",", skip_header=1)[:, :-1]
    training_data = training_data[:, :-1]  # strip an odd string from the data set
    pca.fit(training_data)
    training_data = pca.transform(training_data)
    training_data = training_data[:, 0:50]  # throw away most of the data
    
    testing_labels = np.genfromtxt('test.csv', delimiter=",", usecols=-1, dtype='unicode', skip_header=1, autostrip=True)
    testing_data = np.genfromtxt('test.csv', delimiter=",", skip_header=1)[:, :-1]
    testing_data = testing_data[:, :-1]  # strip an odd string from the data set
    pca.fit(testing_data)
    testing_data = pca.transform(testing_data)
    testing_data = testing_data[:, 0:50]  # throw away most of the data
else:
    # Import Data
    training_labels = np.genfromtxt('train.csv', delimiter = ",", usecols = -1, dtype = 'unicode', skip_header = 1, autostrip = True)
    training_data = np.genfromtxt('train.csv', delimiter = ",", skip_header = 1)[:,:-1]
    training_data = training_data[:,:-1] # strip an odd string from the data set
    training_data = training_data[:, 0:50] # throw away most of the data

    testing_labels = np.genfromtxt('test.csv', delimiter = ",", usecols = -1, dtype = 'unicode', skip_header = 1, autostrip = True)
    testing_data = np.genfromtxt('test.csv', delimiter = ",", skip_header = 1)[:,:-1]
    testing_data = testing_data[:,:-1] # strip an odd string from the data set
    testing_data = testing_data[:, 0:50] # throw away most of the data

# Data structure to hold sorted data
class phoneData:

    def __init__(self, labels, data):
        self.data_dict = {}
        self.mean_vectors = {}  
        self.covariance = {}
        self.prior = {}
        self.total_num_samples = 0
        self.estimated_cov = {}
        
        for i in range(len(labels)):
            if (self.should_class_be_added(labels[i]) == True):
                self.add_class(labels[i])
            self.add_row(labels[i], data[i])   
        
        self.calculate_mean_vector()
        self.calculate_covariances()
        self.calculate_piors()
        
    def add_class(self, label):
        # Each label has its own array of values
        self.data_dict[label] = []
        self.mean_vectors[label] = []
        self.covariance[label] = []
        self.estimated_cov[label] = []

    def add_row(self, label, row):
        self.total_num_samples += 1
        # Stack the new row into our data array
        if len(self.data_dict[label]) >= 1:
            self.data_dict[label] = np.vstack((self.data_dict[label], row))
        else:
            self.data_dict[label] = row
        
    def should_class_be_added(self, label):
        if label in self.data_dict:
            return False
        else:
            return True
            
    def calculate_mean_vector(self):
        current_column = []
        workspace = []
        current_sum = 0
        current_mean = 0
        for label in self.mean_vectors:
            workspace = self.data_dict[label]
            for i in range(len(workspace[0])):
                current_column = workspace[:,i]
                current_avg = np.average(current_column)
                self.mean_vectors[label].append(current_avg)

    def estimate_covariances(self):
        workspace = []
        summation = []
        current_mean_vector = []
        row_minus_mean = []
        proper_dim_row_minus_mean = []
        transpose_row_minus_mean = []
        product = []
        for label in self.data_dict:
            current_mean_vector = self.mean_vectors[label]
            for row in self.data_dict[label]:
                row_minus_mean = np.subtract(row, current_mean_vector)
                proper_dim_row_minus_mean = np.array([row_minus_mean])
                print('row - mean shape', proper_dim_row_minus_mean.shape)
                transpose_row_minus_mean = np.transpose(proper_dim_row_minus_mean)
                print('transpose row - mean shape', transpose_row_minus_mean.shape)
                product = np.matmul(proper_dim_row_minus_mean, transpose_row_minus_mean)
                print('shape of product', product.shape)
                summation = np.add(summation, product)
                
            self.estimated_cov[label] = summation / len(self.data_dict[label][0])
                
        
    def calculate_covariances(self):
        workspace = []
        transpose_workspace = []
        current_cov = []
        for label in self.covariance:
            workspace = self.data_dict[label]
            workspace = workspace
            current_cov = np.cov(workspace, rowvar = False)
            self.covariance[label] = current_cov
    
    def calculate_piors(self):
        num_of_samples_in_this_label = 0
        for label in self.data_dict:
            num_of_samples_in_this_label = len(self.data_dict[label][0])
            self.prior[label] = num_of_samples_in_this_label / self.total_num_samples
            
    def keyWithMaxVal(self, d):
        v = list(d.values())
        k = list(d.keys())
        return k[v.index(max(v))]
            
    def calculate_discriminant(self, data_row):
        current_mean_vector = []
        current_cov = []
        dimension = 0
        row_minus_mean_vector = []
        transpose_row_minus_mean_vector = []
        inverse_cov = []
        det_cov = 0
        temp_discriminant = 0
        discriminant_values = {}
        for label in self.data_dict:
            current_mean_vector = self.mean_vectors[label]
            current_cov = self.covariance[label]
            dimension = len(current_cov)
            row_minus_mean_vector = np.subtract(data_row, current_mean_vector)
            row_minus_mean_vector = np.array([row_minus_mean_vector])
            transpose_row_minus_mean_vector = np.transpose(row_minus_mean_vector)
            det_cov = np.linalg.det(current_cov)
            if det_cov > 0:
                inverse_cov = np.linalg.inv(current_cov)
                first_mat_mul = np.matmul(inverse_cov, transpose_row_minus_mean_vector)
                second_mat_mul = np.matmul(row_minus_mean_vector, first_mat_mul)
                temp_discriminant = (-(dimension / 2) * np.log(2 * np.pi)) - \
                    ((1 / 2) * np.log(det_cov)) - \
                    ((1 / 2) * second_mat_mul) + \
                    (np.log(self.prior[label]))
            else:
                temp_discriminant = 0
                print('Singular Matrix! Results are no good')
            discriminant_values[label] = temp_discriminant
        return self.keyWithMaxVal(discriminant_values)
        
    def getCov(self):
        return self.covariance
        
    def getMean(self):
        return self.mean_vectors
        
    def getPrior(self):
        return self.prior
        
    def show_info(self):
        print('---------Info about data containter---------')
        print('*Number of Classes = ', len(self.data_dict.keys()))
        for j in self.data_dict:
            print('*Info for label ', j)
            print('**Shape of data :',  self.data_dict[j].shape)
            print('**Length of mean vector :', len(self.mean_vectors[j]), ' entries')
            print('**Shape of covariance matrix :', self.covariance[j].shape)
        print('---------Info about data containter---------')


def main():

    train_data = phoneData(training_labels, training_data)
    #train_data.show_info()   

    #test_data = phoneData(testing_labels, testing_data)
    #test_data.show_info()
    
    
    train_total = 0
    train_correct = 0
    for i in range(len(training_labels)):
        train_total += 1
        temp_val = train_data.calculate_discriminant(training_data[i])
        if (temp_val == training_labels[i]):
            train_correct += 1
    print('Training score = ', ((train_correct / train_total) * 100))
    
    test_total = 0
    test_correct = 0
    for i in range(len(testing_labels)):
        test_total += 1
        temp_val = train_data.calculate_discriminant(testing_data[i])
        if (temp_val == testing_labels[i]):
            test_correct += 1
    print('Testing score = ', ((test_correct / test_total) * 100))

if __name__ == '__main__':
    main()
    