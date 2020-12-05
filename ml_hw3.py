
import numpy as np
import matplotlib.pyplot as plt
domain=np.linspace(-10,10,500)
mean = np.zeros(500)
cov = np.identity(500)  
x1, x2, x3, x4 = np.random.multivariate_normal(mean, cov, 4)
plt.plot(domain, x1)
plt.xlim(-10.0, 10.0)
plt.ylim(-3.0, 3.0)
plt.title('random1')
plt.figure()
##
mean = np.zeros(500)
cov = np.ones(250000).reshape(500,500) 
x1, x2, x3, x4 = np.random.multivariate_normal(mean, cov, 4)
plt.plot(domain, x1)
plt.xlim(-10.0, 10.0)
plt.ylim(-3.0, 3.0)
plt.title('one1')
plt.show()
##
mean = np.zeros(500)
cov = [0.5]*250000
cov=np.array(cov).reshape(500,500) 
x1, x2, x3, x4 = np.random.multivariate_normal(mean, cov, 4)
plt.plot(domain, x1)
plt.xlim(-10.0, 10.0)
plt.ylim(-3.0, 3.0)
plt.title('one1')
plt.show()
##6
domain=np.linspace(-10,10,500)
mean = np.zeros(500)
cov = np.ones(250000).reshape(500,500) 
for i in range(cov.shape[0]):
    for j in range(cov.shape[1]):
        cov[i][j]=np.exp(-(domain[i]-domain[j])**2/5)
x1, x2, x3, x4 = np.random.multivariate_normal(mean, cov, 4)
plt.plot(domain, x1)
plt.xlim(-10.0, 10.0)
plt.ylim(-3.0, 3.0)
plt.title('q6.1')
plt.show()
##7
domain=np.linspace(-10,10,500)
mean = np.zeros(500)
cov = np.zeros(250000).reshape(500,500)
for i in range(cov.shape[0]):
    cov[i][i]=1
for i in range(cov.shape[0]):
    for j in range(cov.shape[1]):
        if (i-j)%74==0:
            cov[i][j]=1
x1, x2, x3, x4 = np.random.multivariate_normal(mean, cov, 4)
plt.plot(domain, x1)
plt.xlim(-10.0, 10.0)
plt.ylim(-3.0, 3.0)
plt.title('q7.1')
plt.show()
##9
from numpy.linalg import inv
domain=np.linspace(-10,10,500)
mean = np.zeros(500)
cov500 = np.zeros(250000).reshape(500,500)
cov3=np.zeros(9).reshape(3,3)
cov12=np.zeros(1500).reshape(500,3)
x=[-6,0,7]
y=[3,-2,2]
for i in range(cov12.shape[0]):
    for j in range(cov12.shape[1]):
        cov12[i][j]=np.exp(-(domain[i]-x[j])**2/5)
for i in range(cov500.shape[0]):
    for j in range(cov500.shape[1]):
        cov500[i][j]=np.exp(-(domain[i]-domain[j])**2/5)
cov3=np.zeros(9).reshape(3,3)
for i in range(cov3.shape[0]):
    for j in range(cov3.shape[1]):
        cov3[i][j]=np.exp(-(x[i]-x[j])**2/5)
b=np.dot(np.dot(cov12,inv(cov3)),y)
A=cov500-np.dot(np.dot(cov12,inv(cov3)),cov12.T)
x1, x2, x3, x4 = np.random.multivariate_normal(b, A, 4)
plt.plot(domain, x1)
plt.scatter(x,y)
plt.xlim(-10.0, 10.0)
plt.ylim(-3.0, 3.0)
plt.title('q9.1')
plt.show()
##10
cov = np.zeros(250000).reshape(500,500)
for i in range(cov.shape[0]):
    cov[i][i]=1
for i in range(cov.shape[0]):
    for j in range(cov.shape[1]):
        if (i-j)%74==0:
            cov[i][j]=1
            
b=np.dot(np.dot(cov12,inv(cov3)),y)
x1, x2, x3, x4 = np.random.multivariate_normal(b, cov, 4)
plt.plot(domain, x1)
plt.plot(x,y,'rx')
plt.xlim(-10.0, 10.0)
plt.ylim(-3.0, 3.0)
plt.title('q10.1')
plt.show()

