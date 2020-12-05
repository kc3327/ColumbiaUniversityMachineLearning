from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random as rd

mat = loadmat('mnist_digits.mat')

Data_Y = np.asarray(mat["Y"],dtype= 'float')
Data_X = np.asarray(mat["X"],dtype= 'float')

X, X_test = train_test_split(Data_X, test_size=0.3, random_state = 88)
Y, Y_test = train_test_split(Data_Y, test_size=0.3, random_state = 88)

def init_w(data):            #Initialized the w0 
    a, b = np.shape(data)
    w0 = np.zeros([b,1])
    return w0

def one_num(labels, digit):  #Categorize the labels into target label and the rest
    if labels == digit:
        return 1
    else: 
        return -1

def V0_train_one(data, labels, digit, iterations = 1): # Train the weight for a single chosen digit
    w0 = init_w(data).T
    for i in range(0,iterations):   
        for j in range(len(data)):   #For each row vector, find out the row vector with wrong prediction and add/substract it to the weight vector
            tag = one_num(labels[j], digit)
            dotproduct = np.dot((w0), data[j,:].reshape(784,1))[0][0]
            if  dotproduct * tag  <= 0:
                w0 += data[j] * tag
    return w0
            
def V0_training_all(data, labels, iterations = 3): # Train weights for all the digits from 0 to 9
    Weight = []
    for i in range(0,10):
        Weight.append(V0_train_one(data, labels, i, iterations))
    return Weight

def Test(data, labels, Weight): #return the dataframe of true label and predicted label in test data 
    df = pd.DataFrame(pd.DataFrame(np.concatenate(Weight)))
    df_test = pd.DataFrame(pd.DataFrame(np.concatenate(labels))) #Create test dataframe
    df_test['predict'] = 0
    for i in range(len(data)):
        b = df.dot(pd.Series(data[i])).tolist()
        df_test.iloc[i,1] = b.index(max(b)) #Assigning the predicted digit for each data point
    return df_test
##Note: The test function could be applied to both V0 and V1

V0_Weight = V0_training_all(X,Y,4)
df_result = Test(X_test, Y_test, V0_Weight) #Training Weights
print(df_result[df_result[0] == df_result['predict']].shape[0]/3000)#Return Accuracy Rate
