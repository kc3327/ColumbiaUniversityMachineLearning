import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
import heapq
import scipy as scp
import scipy.stats as ss
from sklearn.decomposition import PCA
swiss_roll_hole=pd.read_csv("swiss_roll_hole.txt")
# swiss_roll_hole.columns=['x1','x2','x3']
swiss_roll=pd.read_csv("swiss_roll.txt",header=None)
# swiss_roll.columns=['x1','x2','x3']

dic={}

for ch in range(len(swiss_roll)):
    a=swiss_roll.iloc[ch][0]
    temp=np.array([float(a.split()[0]),float(a.split()[1]),float(a.split()[2])])
    dic[ch]=temp


knn_dic={}
for i in range(len(distance)):
    g=[np.linalg.norm(dic[i]-dic[x]) for x in range(len(swiss_roll))]
    knn_dic[i]=np.array(g).argsort()[:10]
    for ch in np.array(g).argsort()[:10]:
        distance[i][ch]=np.linalg.norm(dic[i]-dic[ch])

path=distance.copy()
##reference:https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-greedy-algo-7/
def minDistance(dist, sptSet): 

    # Initilaize minimum distance for next node 
    min = np.inf
    min_index=0
    # Search not nearest vertex not in the  
    # shortest path tree 
    for v in range(4000): 
        if dist[v] < min and sptSet[v] == False: 
            min = dist[v] 
            min_index = v 

    return min_index



for i in range(len(swiss_roll)):
    unvisited=[False for x in range(len(swiss_roll))]
    unvisited[i]=True    
    for z in range(len(swiss_roll)):
        u=minDistance(path[i], unvisited)
        unvisited[u]=True
        for j in knn_dic[u]: ##j= 2 5
            if path[u][j]+path[i][u]<path[i][j]:
                path[i][j]=path[u][j]+path[i][u]  
                
                
def Gradient_Descent(x,y,maxtime=1000,learning_rate=0.005,tolerance=0.1):
    thet=np.array([1]*x.shape[0])
    
    m=len(y)
    loss_history=np.zeros(maxtime)
    theta_history=np.zeros((maxtime,X.shape[0]))
    for i in range(maxtime):
#         print(thet)
        prediction=np.dot(thet,x)
        thet=thet-learning_rate*2/m*np.dot((prediction-y),x.T)
        theta_history[i,:]=thet
        loss_history[i]=loss_function(X,Y,thet)
        if loss_function(X,Y,thet)<=tolerance:
            break
    return thet,loss_history,theta_history,i

pca = PCA(n_components=2)
pca.fit(df)
r = pca.transform(df)

