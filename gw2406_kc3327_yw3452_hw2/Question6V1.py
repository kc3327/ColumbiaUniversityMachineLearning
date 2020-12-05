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

def V1_train_one(data, labels, digit, iterations = 1): #Train weight for one digit
    T = iterations * data.shape[0]
    t=0
    w0 = init_w(data).T
    dot_product = np.dot(data,w0.T) # Initialize dot product of w0*x_i
    tag = np.array([[one_num(x, digit) for x in labels]]).T #Categorize the labels into target digit and the rest
    
    while (min(dot_product * tag) <= 0)[0] and (t <= T): #Set up conditions for running the algorithm
        product = dot_product * tag
        row = np.where(product == np.min(product))[0][0] #Retrieve the row index of minimum value of y*w*x
        if tag[row] * (np.dot(data[row], w0.T)) <= 0: #For minimum value, if negative, add/substract the row vector to the weight vector
            w0 += data[row].reshape(1,784) *  tag[row] #Update weight vector
            dot_product = np.dot(data,w0.T) #Update w*x
        t += 1
    return w0

def V1_training_all(data, labels, iterations = 3): #Train the weights for all digits 
    Weight = []
    for i in range(0,10):
        Weight.append(V1_train_one(data, labels, i, iterations)) #Classification
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
V1_Weight = V1_training_all(X, Y, iterations = 3) 
df_result = Test(X_test, Y_test, V1_Weight)
print(df_result[df_result[0] == df_result['predict']].shape[0]/3000) # Return Accuracy Rate

