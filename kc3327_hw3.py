import numpy as np
import seaborn as sns
M=np.array([x for x in range(1,26)])
M.resize(5,5)
map_dic={}
for i in range(5):
    for j in range(5):
        if M[i,j] not in map_dic:
            map_dic[M[i,j]]=[]
        try:
            map_dic[M[i,j]].append(M[i,j+1])##right
        except IndexError:
            map_dic[M[i,j]].append(M[i,j])
        try:
            if j==0:
                map_dic[M[i,j]].append(M[i,j])
            else:    
                map_dic[M[i,j]].append(M[i,j-1])##left
        except IndexError:
            map_dic[M[i,j]].append(M[i,j])
        try:
            map_dic[M[i,j]].append(M[i+1,j])##down
        except IndexError:
            map_dic[M[i,j]].append(M[i,j])
        try:
            if i==0:
                map_dic[M[i,j]].append(M[i,j])
            else:
                map_dic[M[i,j]].append(M[i-1,j])##up
        except IndexError:
            map_dic[M[i,j]].append(M[i,j])
map_dic[2]=[22,22,22,22]
map_dic[4]=[14,14,14,14]

gamma=0.5
A=np.zeros([25,25])
for i in range(25):
    for j in range(25):
        if i==j:
            A[i,j]=gamma*map_dic[i+1].count(i+1)/4-1
        else:
            if j+1 in map_dic[i+1]:
                A[i,j]=gamma*map_dic[i+1].count(j+1)/4  

reward=np.zeros([25,25])
for i in range(25):
    for j in range(25):
        if i==j:
            reward[i,j]=-1
reward[1,21]=10
reward[3,13]=5

b=np.zeros(25)
for i in range(25):
        up=[1  if map_dic[i+1][3]==z else 0 for z in range(1,26)]
        right=[1  if map_dic[i+1][0]==z else 0 for z in range(1,26)]
        left=[1  if map_dic[i+1][1]==z else 0 for z in range(1,26)]
        down=[1  if map_dic[i+1][2]==z else 0 for z in range(1,26)]
        b[i]=-(np.dot(reward[i],up)+np.dot(reward[i],right)+np.dot(reward[i],down)+np.dot(reward[i],left))/4
result=np.linalg.solve(A,b)
pic1 = sns.heatmap(result.reshape(5,5),annot = True)
gamma=0.5
A2=np.zeros([25,25])
for i in range(25):
    for j in range(25):
        if i==j:
            up=[1  if map_dic[i+1][3]==j+1 else 0]
            right=[1  if map_dic[i+1][0]==j+1 else 0]
            left=[1  if map_dic[i+1][1]==j+1 else 0]
            down=[1  if map_dic[i+1][2]==j+1 else 0]
            A2[i,j]=gamma*(sum(up)*0.7+sum(right)*0.1+sum(left)*0.1+sum(down)*0.1)-1
        else:
            if j+1 in map_dic[i+1]:
                up=[1  if map_dic[i+1][3]==j+1 else 0]
                right=[1  if map_dic[i+1][0]==j+1 else 0]
                left=[1  if map_dic[i+1][1]==j+1 else 0]
                down=[1  if map_dic[i+1][2]==j+1 else 0]         
                A2[i,j]=gamma*(sum(up)*0.7+sum(right)*0.1+sum(left)*0.1+sum(down)*0.1)

b2=np.zeros(25)
for i in range(25):
    up=[1  if map_dic[i+1][3]==z else 0 for z in range(1,26)]
    right=[1  if map_dic[i+1][0]==z else 0 for z in range(1,26)]
    left=[1  if map_dic[i+1][1]==z else 0 for z in range(1,26)]
    down=[1  if map_dic[i+1][2]==z else 0 for z in range(1,26)]
    b2[i]=-(0.7*np.dot(reward[i],up)+0.1*np.dot(reward[i],right)+0.1*np.dot(reward[i],down)+0.1*np.dot(reward[i],left))  
result1=np.linalg.solve(A2,b2)
pic2 = sns.heatmap(result1.reshape(5,5),annot = True)
gamma=0.5
V=np.zeros(25)
M=[0 for x in range(25)]
for j in range(1000):
    for i in range(25):
        action=np.zeros(4)
        up=[1  if map_dic[i+1][3]==z else 0 for z in range(1,26)]
        right=[1  if map_dic[i+1][0]==z else 0 for z in range(1,26)]
        left=[1  if map_dic[i+1][1]==z else 0 for z in range(1,26)]
        down=[1  if map_dic[i+1][2]==z else 0 for z in range(1,26)]
        action[0]=np.dot(reward[i]+gamma*V,up)
        action[1]=np.dot(reward[i]+gamma*V,down)
        action[2]=np.dot(reward[i]+gamma*V,right)
        action[3]=np.dot(reward[i]+gamma*V,left)
        V[i]=max(action)
        if np.argmax(action)==0:
            M[i]="up"
        elif np.argmax(action)==1:
            M[i]="down"
        elif np.argmax(action)==2:
            M[i]="right"
        else:
            M[i]="left"
print(np.array(M).reshape(5,5))
pic3 = sns.heatmap(np.array(V).reshape(5,5),annot = True)
