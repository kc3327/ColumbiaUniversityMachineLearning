
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dropout
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, LeakyReLU, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2
import tensorflow as tf
import glob
import os
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier

seed = 256
np.random.seed(seed)
tf.random.set_seed(seed)

#%%
path1 = '/Users/alun/Desktop/Columbia/Spring2020/Machine Learning/KaggleProject/4771-sp20-covid/train/train'
path2='/Users/alun/Desktop/Columbia/Spring2020/Machine Learning/KaggleProject/4771-sp20-covid/train.csv'
trainX,trainY,trainY_uncategorical=get_clean_train_features_label(path1,path2)

path3 = '/Users/alun/Desktop/Columbia/Spring2020/Machine Learning/KaggleProject/4771-sp20-covid/test/test'
testX=get_test_features(path3)



#%% for cnn
def get_clean_train_features_label(path_x,path_y):
    
    filename_train=[]
    for filename in glob.glob(os.path.join(path_x, '*.jpeg')):
        filename_train.append(filename)
    print(len(filename_train))
    filename_train.sort(key= lambda x: int(x.split('/')[-1].split('-')[-1].split('.')[0]))
    
    trainX=np.array([])
    t=0
    for f in filename_train:
        if t==0:
            img = load_img(f,target_size=(256, 256))  
            x = img_to_array(img)[20:230,20:230,:]
            temp=array_to_img(x).resize((256,256))
            x=img_to_array(temp)
            trainX = x.reshape((1,) + x.shape)
            t+=1
        else:
            img = load_img(f,target_size=(256, 256))  
            x = img_to_array(img)[20:230,20:230,:]
            temp=array_to_img(x).resize((256,256))
            x=img_to_array(temp)
            x = x.reshape((1,) + x.shape)
            trainX=np.append(trainX,x,axis=0)
    df_testY=pd.read_csv(path_y)
    match_dic={'normal':0, 'viral':1, 'bacterial':2, 'covid':3}
    trainY_unca=np.array(df_testY['label'].apply(lambda x:match_dic[x]))
   
    trainY=to_categorical(trainY_unca)
    return trainX,trainY,trainY_unca


def get_test_features(path_x):
    filename_train=[]
    for filename in glob.glob(os.path.join(path_x, '*.jpeg')):
        filename_train.append(filename)
    print(len(filename_train))
    filename_train.sort(key= lambda x: int(x.split('/')[-1].split('-')[-1].split('.')[0]))
    
    trainX=np.array([])
    t=0
    for f in filename_train:
        if t==0:
            img = load_img(f,target_size=(256, 256))  
            x = img_to_array(img)[20:230,20:230,:]
            temp=array_to_img(x).resize((256,256))
            x=img_to_array(temp)
            trainX = x.reshape((1,) + x.shape)
            t+=1
        else:
            img = load_img(f,target_size=(256, 256))  
            x = img_to_array(img)[20:230,20:230,:]
            temp=array_to_img(x).resize((256,256))
            x=img_to_array(temp)
            x = x.reshape((1,) + x.shape)
            trainX=np.append(trainX,x,axis=0)
    return trainX

#%%
def get_data(path_x,path_y):
    
    filename_train=[]
    for filename in glob.glob(os.path.join(path_x, '*.jpeg')):
        filename_train.append(filename)
    print(len(filename_train))
    filename_train.sort(key= lambda x: int(x.split('/')[-1].split('-')[-1].split('.')[0]))
    trainX=[]
    for f in filename_train:
        x=np.asarray(Image.open(f).resize((256,256)))[10:240,10:240].flatten()
        trainX.append(x)
    df_testY=pd.read_csv(path_y)
    match_dic={'normal':0, 'viral':1, 'bacterial':2, 'covid':3}
    trainY_unca=np.array(df_testY['label'].apply(lambda x:match_dic[x]))

    return trainX,trainY_unca


def get_test_feature(path_x):
    
    filename_train=[]
    for filename in glob.glob(os.path.join(path_x, '*.jpeg')):
        filename_train.append(filename)
    print(len(filename_train))
    filename_train.sort(key= lambda x: int(x.split('/')[-1].split('-')[-1].split('.')[0]))
    trainX=[]
    for f in filename_train:
        x=np.asarray(Image.open(f).resize((256,256)))[10:240,10:240].flatten()
        trainX.append(x)
    return trainX
#%%
def generate_model():
    cnn = Sequential()
    cnn.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',padding='same',kernel_regularizer=l2(0.001),input_shape=(256, 256, 3)))
    cnn.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',padding='same',kernel_regularizer=l2(0.001)))
    cnn.add(MaxPooling2D((2, 2)))
    cnn.add(Dropout(0.1))
    cnn.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform',padding='same',kernel_regularizer=l2(0.001)))
    cnn.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform',padding='same',kernel_regularizer=l2(0.001)))
    cnn.add(MaxPooling2D((2, 2)))
    cnn.add(Dropout(0.1))
    cnn.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform',padding='same',kernel_regularizer=l2(0.001)))
    cnn.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform',padding='same',kernel_regularizer=l2(0.001)))
    cnn.add(MaxPooling2D((2, 2)))
    cnn.add(Dropout(0.1))
    cnn.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform',padding='same',kernel_regularizer=l2(0.001)))
    cnn.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform',padding='same',kernel_regularizer=l2(0.001)))
    cnn.add(MaxPooling2D((2, 2)))
    cnn.add(Dropout(0.1))
    cnn.add(Flatten())
    cnn.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
    cnn.add(Dense(4, activation='softmax'))
    cnn.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    return cnn


#trainX,testX=trainX/255,testX/255

cnn=generate_model()
img_generator=ImageDataGenerator(rotation_range=5,rescale=1./255,width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, vertical_flip=True)
data_aug_train = img_generator.flow(trainX, trainY, batch_size=64)
cnnfit = cnn.fit_generator(data_aug_train, steps_per_epoch=int(trainX.shape[0] / 64), epochs=100)


#%% data for traditional algorithmns
path1 = '/Users/alun/Desktop/Columbia/Spring2020/Machine Learning/KaggleProject/4771-sp20-covid/train/train'
path2='/Users/alun/Desktop/Columbia/Spring2020/Machine Learning/KaggleProject/4771-sp20-covid/train.csv'
train_X,train_Y=get_data(path1,path2)
path3 = '/Users/alun/Desktop/Columbia/Spring2020/Machine Learning/KaggleProject/4771-sp20-covid/test/test'
test_X=get_test_feature(path3)
train_X,test_X=[x/255 for x in train_X],[x/255 for x in test_X]
X_train, X_test, y_train, y_test=train_test_split(train_X,train_Y,test_size=0.2, random_state=256)




#%% SVM

sgd= SGDClassifier(max_iter=10000, tol=1e-3)
sgd.fit(X_train, y_train)
y_predict=sgd.predict(X_test)
print(accuracy_score(y_predict,y_test))

def output_csv(y_predict):
    match_dic_inverse={0:'normal', 1:'viral', 2:'bacterial', 3:'covid'}
    final_result=[match_dic_inverse[x] for x in y_predict]
    df=pd.read_csv('/Users/alun/Desktop/Columbia/Spring2020/Machine Learning/KaggleProject/4771-sp20-covid/submission.csv')
    df['label']=final_result
    df=df.set_index('Id')
    df.to_csv('/Users/alun/Desktop/Columbia/Spring2020/Machine Learning/KaggleProject/submission_12.csv')


#%% random forest
from sklearn.ensemble import RandomForestClassifier
randomforest=RandomForestClassifier(n_estimators=500,criterion='entropy',max_features='auto',random_state=256)
randomforest.fit(X_train, y_train)
y_predict=randomforest.predict(X_test)
print(accuracy_score(y_predict,y_test))
cf=confusion_matrix(y_predict,y_test)
import seaborn as sns
labels=['normal', 'viral', 'bacterial', 'covid']
ax1=sns.heatmap(confusion_matrix(y_predict,y_test),annot=True,xticklabels=labels,yticklabels=labels)


#%% output
import seaborn as sns
feature_importance_matrix=randomforest.feature_importances_.reshape((256,256))
ax=sns.heatmap(feature_importance_matrix)
rf_test_label=randomforest.predict(test_X)
output_csv(rf_test_label)

#%%
path='/Users/alun/Desktop/Columbia/Spring2020/Machine Learning/KaggleProject/submission2.csv'
df=pd.read_csv(path)
match_dic={'normal':0, 'viral':1, 'bacterial':2, 'covid':3}
result=np.array(df['label'].apply(lambda x: match_dic[x]))

#%%
loaded_model = tf.keras.models.load_model('/Users/alun/Desktop/Columbia/Spring2020/Machine Learning/KaggleProject/fit_epochs100_validation.h5') 
loaded_model.summary()

path1 = '/Users/alun/Desktop/Columbia/Spring2020/Machine Learning/KaggleProject/4771-sp20-covid/train/train'
path2='/Users/alun/Desktop/Columbia/Spring2020/Machine Learning/KaggleProject/4771-sp20-covid/train.csv'
trainX,trainY,c=get_clean_train_features_label(path1,path2)



#%%
from sklearn.metrics import confusion_matrix
a,testf,b,testl=train_test_split(trainX,trainY,random_state=300,train_size=0.8)
yp=loaded_model.predict(testf/255)

testlabel=[]
for ch in testl:
    testlabel.append(np.argmax(ch))
testtemp=[]
for ch in yp:
    testtemp.append(np.argmax(ch))
cf=confusion_matrix(testlabel,testtemp)
import seaborn as sns
labels=['normal', 'viral', 'bacterial', 'covid']
ax1=sns.heatmap(confusion_matrix(testlabel,testtemp),annot=True,xticklabels=labels,yticklabels=labels)
print(accuracy_score(testlabel,testtemp))



#%%best

path3 = '/Users/alun/Desktop/Columbia/Spring2020/Machine Learning/KaggleProject/4771-sp20-covid/test/test'
testXX=get_test_features(path3)
y_predict=loaded_model.predict(testXX/255)
#%%
match_dic_inverse={0:'normal', 1:'viral', 2:'bacterial', 3:'covid'}
final_result=[]
for ch in y_predict:
  final_result.append(match_dic_inverse[np.argmax(ch)])

df=pd.read_csv('/Users/alun/Desktop/Columbia/Spring2020/Machine Learning/KaggleProject/4771-sp20-covid/submission.csv')
df['label']=final_result
df=df.set_index('Id')
df.to_csv('/Users/alun/Desktop/Columbia/Spring2020/Machine Learning/KaggleProject/submission_13.csv')









