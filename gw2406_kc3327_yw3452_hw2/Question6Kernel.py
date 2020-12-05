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

def polynomial_kernel(x, y, p=2):  #Kernelization
    return (np.dot(x, y)) ** p

def Kernel_train_one(data, labels, digit, iterations = 2, degree = 7): #Train alpha for 1 digit
    alpha = np.zeros(X.shape[0]).reshape(X.shape[0],1) #Initialize alpha
    tag = np.array([[one_num(x, digit) for x in labels]]).T #Classify between chosen digit and the rest
    t = 1
    while t <= iterations:
        for j in range(len(data)):
            if np.sign(np.dot(polynomial_kernel(data,data[j], degree),alpha * tag)) != tag[j]: #Identify rows with mistakes
                alpha[j] += 1
        t += 1
    return alpha

def Kernel(data, labels,data_test, labels_test,iterations = 1, degree = 5): #Return dataframe of true digits and predicted digits
    alpha = []
    
    for i in range(0,10): #Train all
        alpha.append(Kernel_train_one(data, labels, i, iterations, degree))
        
    df_test = pd.DataFrame(pd.DataFrame(np.concatenate(labels_test)))
    df_test['predict'] = 0
    
    for j in range(len(data_test)): #Testing
        b = []
        
        for k in range(0,10):
            tag = np.array([[one_num(x, k) for x in labels]]).T 
            b.append(np.dot(polynomial_kernel(data,data_test[j],degree), alpha[k] * tag)) #Append the sum of alpha*x*y to the list
            
        df_test.iloc[j,1] = b.index(max(b))#Classify the digit with highest product value as the predicted digit
        
    return df_test

Kernel_result = Kernel(X,Y,X_test,Y_test)
print(Kernel_result[Kernel_result[0] == Kernel_result['predict']].shape[0]/3000)  #Prediction Accuracy
