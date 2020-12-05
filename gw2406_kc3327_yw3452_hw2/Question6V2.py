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
    
def V2_train_one(data, labels, digit, iterations=1): #Train weight for one digit
    w0 = init_w(data).T
    w_array = w0 #Initialize the weight list 
    c = 0 
    c_list = [c] #Initialize the c list
    for i in range(iterations):
        for j in range(len(data)):
            y = one_num(labels[j], digit)
            dot_product = np.dot(w0,data[j,:].reshape(784,1))[0][0]
            if y * dot_product <= 0:
                w0 += y*data[j] #Add/Substract data vector to weight vector
                w_array = np.concatenate((w_array,w0)) #Proceed to new weight vector if y*w*x is negative
                c_list.append(1) #Proceed to new c if y*w*x is negative
            else:
                c_list[-1] += 1 #Increasing the current c value by 1 if next prediction is correct 
    return (w_array, np.asarray(c_list))

def V2_training_all(data, labels, iterations = 1): #Train weights for all digits
    Weight = []
    for i in range(0,10):
        Weight.append(V2_train_one(data, labels, i, iterations)) #Create list by appending weight lists and c lists of each digits
    return Weight

def Test_V2(data, labels, Weight):
    df_test = pd.DataFrame(pd.DataFrame(np.concatenate(labels)))
    df_test['predict'] = 0
    b = []
    for i in range(len(data)): #Classification
        b = []
        for j in range(10):
            b.append(np.dot(np.dot(Weight[j][0],X_test[i]),Weight[j][1]))
            df_test.iloc[i,1] = b.index(max(b)) #Classify the digit with highest product value as the predicted digit
    return df_test

V2_Weight = V2_training_all(X, Y, iterations = 4)
df_result =Test_V2(X_test,Y_test,V2_Weight)
print(df_result[df_result[0] == df_result['predict']].shape[0]/3000)  #Prediction Accuracy
