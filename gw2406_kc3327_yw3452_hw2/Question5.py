import numpy as np
import matplotlib.pyplot as plt
import math
np.random.seed(2)
theta=np.array([4,2,3,4,5,6,7])
X = 10*np.random.rand(1000,6)#two dimensional data set
constant=np.array([1]*1000)
constant=constant.reshape(1000,1)
X= (np.hstack(( constant,X))+np.random.randn(1000,1)).T
Y = np.dot(theta,X)
def loss_function(X,Y,thet):
    return 1/len(Y)*np.sum(np.square(Y-np.dot(thet,X)))   
def Gradient_Descent(x,y,maxtime=1000,learning_rate=0.005,tolerance=0.1):
    thet=np.array([1]*x.shape[0])
    
    m=len(y)
    loss_history=np.zeros(maxtime)
    theta_history=np.zeros((maxtime,X.shape[0]))
    for i in range(maxtime):
        prediction=np.dot(thet,x)
        thet=thet-learning_rate*2/m*np.dot((prediction-y),x.T)
        theta_history[i,:]=thet
        loss_history[i]=loss_function(X,Y,thet)
        if loss_function(X,Y,thet)<=tolerance:
            break
    return thet,loss_history,theta_history,i
def Stochastic_Gradient_Descent(x,y,minibatch=32,tolerance=0.1,maxtime=1000,learning_rate=0.005):
    thet=np.array([1]*x.shape[0])
    m=len(y)
    loss_history=np.zeros(maxtime)
    theta_history=np.zeros((maxtime,x.shape[0]))
    max_poss=int((math.factorial(m))/(math.factorial(m-minibatch)*math.factorial(minibatch)))
    for i in range(max_poss):
        if i >=maxtime:
            break
        random_selection=np.random.choice(m, minibatch, replace=False)
        x_selection=x[:,random_selection]
        y_selection=y[random_selection]
        prediction=np.dot(thet,x_selection)
        thet=thet-learning_rate*2/minibatch*np.dot((prediction-y_selection),x_selection.T)
        theta_history[i,:]=thet
        loss_history[i]=loss_function(X,Y,thet)
        if loss_function(X,Y,thet)<=tolerance:
            break
    return thet,loss_history,theta_history,i

def Stochastic_Gradient_Descent_Momentum(x,y,minibatch=32,del_thet=np.array([0]),alpha=0.1,tolerance=0.1,maxtime=1000,learning_rate=0.005):
    thet=np.array([1]*x.shape[0])
    m=len(y)
    loss_history=np.zeros(maxtime)
    theta_history=np.zeros((maxtime,x.shape[0]))
    max_poss=int((math.factorial(m))/(math.factorial(m-minibatch)*math.factorial(minibatch)))
    for i in range(max_poss):
        if i >=maxtime:
            break
        random_selection=np.random.choice(m, minibatch, replace=False)
        x_selection=x[:,random_selection]
        y_selection=y[random_selection]
        prediction=np.dot(thet,x_selection)
        del_thet=alpha*del_thet-learning_rate*2/minibatch*np.dot((prediction-y_selection),x_selection.T)
        thet=thet+del_thet
        theta_history[i,:]=thet
        loss_history[i]=loss_function(X,Y,thet)
        if loss_function(X,Y,thet)<=tolerance:
            break
    return thet,loss_history,theta_history,i    

def Stochastic_Gradient_Descent_AdaGrad(x,y,minibatch=32,tolerance=0.1,maxtime=1000,learning_rate=5):
    s=np.array([0]*x.shape[0])
    thet=np.array([1]*x.shape[0])
    m=len(y)
    loss_history=np.zeros(maxtime)
    theta_history=np.zeros((maxtime,x.shape[0]))
    max_poss=int((math.factorial(m))/(math.factorial(m-minibatch)*math.factorial(minibatch)))
    for i in range(max_poss):
        if i >=maxtime:
            break
        random_selection=np.random.choice(m, minibatch, replace=False)
        x_selection=x[:,random_selection]
        y_selection=y[random_selection]
        prediction=np.dot(thet,x_selection)
        g=2/minibatch*np.dot((prediction-y_selection),x_selection.T)
        s=s+g*g
        thet=thet-learning_rate/np.sqrt(s)*g
        theta_history[i,:]=thet
        loss_history[i]=loss_function(X,Y,thet)
        if loss_function(X,Y,thet)<=tolerance:

            break
    return thet,loss_history,theta_history,i
def Stochastic_Gradient_Descent_RMSProp(x,y,minibatch=32,gamma=0.9,tolerance=0.1,maxtime=1000,learning_rate=0.05):
    m=len(y)
    thet=np.array([1]*x.shape[0])
    g_mean=np.array([0]*x.shape[0])
    loss_history=np.zeros(maxtime)
    theta_history=np.zeros((maxtime,x.shape[0]))
    max_poss=int((math.factorial(m))/(math.factorial(m-minibatch)*math.factorial(minibatch)))
    for i in range(max_poss):
        if i >=maxtime:
            break
        random_selection=np.random.choice(m, minibatch, replace=False)
        x_selection=x[:,random_selection]
        y_selection=y[random_selection]
        prediction=np.dot(thet,x_selection)
        g=2/minibatch*np.dot((prediction-y_selection),x_selection.T)
        
        g_mean=gamma*g_mean+(1-gamma)*np.square(g)
        thet=thet-learning_rate/np.sqrt(g_mean+0.000001)*g
      
           
        
        theta_history[i,:]=thet
        loss_history[i]=loss_function(X,Y,thet)
        if loss_function(X,Y,thet)<=tolerance:
            break
    return thet,loss_history,theta_history,i
def Adam(x,y,minibatch=32,beta1=0.9,beta2=0.999,tolerance=0.1,maxtime=1000,learning_rate=10):
    m=len(y)
    thet=np.array([1]*x.shape[0])
    r=np.array([0]*x.shape[0])
    v=np.array([0]*x.shape[0])
    loss_history=np.zeros(maxtime)
    theta_history=np.zeros((maxtime,x.shape[0]))
    max_poss=int((math.factorial(m))/(math.factorial(m-minibatch)*math.factorial(minibatch)))
    for i in range(max_poss):
        if i >=maxtime:
            break
        random_selection=np.random.choice(m, minibatch, replace=False)
        x_selection=x[:,random_selection]
        y_selection=y[random_selection]
        prediction=np.dot(thet,x_selection)
        g=2/minibatch*np.dot((prediction-y_selection),x_selection.T)
        r=beta1*r+(1-beta1)*g
        v=beta2*v+(1-beta2)*np.square(g)
        r_hat=r/(1-np.power(beta1,i+1)+0.0000001)
        v_hat=v/(1-np.power(beta2,i+1)+0.0000001)
        thet=thet-learning_rate*r_hat/np.sqrt(v_hat+0.000000001)
        theta_history[i,:]=thet
        loss_history[i]=loss_function(X,Y,thet)
        if loss_function(X,Y,thet)<=tolerance:
            break
    return thet,loss_history,theta_history,i
def AdaDelta(x,y,minibatch=32,rho=0.999,tolerance=0.1,maxtime=1000,learning_rate=10):
    g_mean=np.array([0.01]*x.shape[0])
    thet=np.array([1]*x.shape[0])
    del_thet_sqr_mean=np.array([10]*x.shape[0])
    m=len(y)
    loss_history=np.zeros(maxtime)
    theta_history=np.zeros((maxtime,x.shape[0]))
    max_poss=int((math.factorial(m))/(math.factorial(m-minibatch)*math.factorial(minibatch)))
    for i in range(max_poss):
        if i >=maxtime:
            break
        random_selection=np.random.choice(m, minibatch, replace=False)
        x_selection=x[:,random_selection]
        y_selection=y[random_selection]
        prediction=np.dot(thet,x_selection)
        
        g=2/minibatch*np.dot((prediction-y_selection),x_selection.T)
#         g_mean=(g_mean*i+np.square(g))/(i+1)
        g_mean=rho*g_mean+(1-rho)*np.square(g)
        del_thet= 0-np.sqrt(del_thet_sqr_mean+0.00001)/(np.sqrt(g_mean+0.00001))*g
#         del_thet_sqr_mean=(del_thet_sqr_mean*i+np.square(del_thet))/(i+1)
        del_thet_sqr_mean=rho*del_thet_sqr_mean+(1-rho)*np.square(del_thet)
        thet=thet+del_thet
        theta_history[i,:]=thet
        loss_history[i]=loss_function(X,Y,thet)
        if loss_function(X,Y,thet)<=tolerance:
            break
    return thet,loss_history,theta_history,i

def plot_loss(n,y,tit,line_color,i):
    print(tit)
    y1=y[:n]
    x=np.linspace(1,n,n)
    

    plt.figure(figsize=(10, 10), dpi=80)
    plt.title(tit)
    plt.grid(True)


    plt.xlabel("Iteration")
    plt.xlim(0, 300)
    plt.xticks(np.linspace(0, n, 21))

    plt.ylabel("Loss Function")

    plt.plot(x, y1, line_color, linewidth=1.0, label="loss")


    plt.legend(loc="upper left", shadow=True)
    plt.plot([i], [0],line_color, marker='o', markersize=10)


    plt.show()
    
def plot_all(n):

    y1,i1=Gradient_Descent(X,Y,maxtime=1000)[1],Gradient_Descent(X,Y,maxtime=1000)[3]

    y2,i2=Stochastic_Gradient_Descent(X,Y,maxtime=1000)[1],Stochastic_Gradient_Descent(X,Y,maxtime=1000)[3]
#     plot_loss(f,y,'Stochastic_Gradient_Descent','g-',i)

    y3,i3=Stochastic_Gradient_Descent_Momentum(X,Y,maxtime=1000)[1],Stochastic_Gradient_Descent_Momentum(X,Y,maxtime=1000)[3]
#     plot_loss(f,y,'Stochastic_Gradient_Descent_Momentum','b-',i)
    y4,i4=Stochastic_Gradient_Descent_AdaGrad(X,Y,maxtime=1000)[1],Stochastic_Gradient_Descent_AdaGrad(X,Y,maxtime=1000)[3]
#     plot_loss(f,y,'Stochastic_Gradient_Descent_AdaGrad','y-',i)
    y5,i5=Stochastic_Gradient_Descent_RMSProp(X,Y,maxtime=1000)[1],Stochastic_Gradient_Descent_RMSProp(X,Y,maxtime=1000)[3]
#     plot_loss(f,y,'Stochastic_Gradient_Descent_RMSProp','c-',i)
    y6,i6=Adam(X,Y,maxtime=1000)[1],Adam(X,Y,maxtime=1000)[3]
#     plot_loss(f,y,'Adam','k-',i)    
    y7,i7=AdaDelta(X,Y,maxtime=1000)[1],AdaDelta(X,Y,maxtime=1000)[3]
#     plot_loss(f,y,'AdaDelta','m-',i)
    y1=y1[:n]
    y2=y2[:n]
    y3=y3[:n]
    y4=y4[:n]
    y5=y5[:n]
    y6=y6[:n]
    y7=y7[:n]
    
    
    
    
    x=np.linspace(1,n,n)

    plt.figure(figsize=(10, 10), dpi=80)
    plt.title('All GD')
    plt.grid(True)

    plt.xlabel("Iteration")
    plt.xlim(0, n)
    plt.xticks(np.linspace(0, n, 21))


    plt.ylabel("Loss Function")
    plt.ylim(0, 100.0)

    
    yy=[y1,y2,y3,y4,y5,y6,y7]
    ii=[i1,i2,i3,i4,i5,i6,i7]
    name=['Gradient_Descent','Stochastic_Gradient_Descent','Momentum','AdaGrad','RMSProp','Adam','AdaDelta']
    line_color=['r-','g-','b-','y-','c-','k-','m-']
    for i in range(len(yy)):
        plt.plot(x, yy[i], line_color[i], linewidth=1.0, label=name[i])
        plt.plot(ii[i], [0],line_color[i][0], marker='o', markersize=10)
    plt.legend(loc="upper left", shadow=True)

    plt.show()
    


plot_all(300)

def output(f):
    y,i=Gradient_Descent(X,Y,maxtime=1000)[1],Gradient_Descent(X,Y,maxtime=1000)[3]
    plot_loss(f,y,'Gradient_Descent','r-',i)
    y,i=Stochastic_Gradient_Descent(X,Y,maxtime=1000)[1],Stochastic_Gradient_Descent(X,Y,maxtime=1000)[3]
    plot_loss(f,y,'Stochastic_Gradient_Descent','g-',i)
    y,i=Stochastic_Gradient_Descent_Momentum(X,Y,maxtime=1000)[1],Stochastic_Gradient_Descent_Momentum(X,Y,maxtime=1000)[3]
    plot_loss(f,y,'Stochastic_Gradient_Descent_Momentum','b-',i)
    y,i=Stochastic_Gradient_Descent_AdaGrad(X,Y,maxtime=1000)[1],Stochastic_Gradient_Descent_AdaGrad(X,Y,maxtime=1000)[3]
    plot_loss(f,y,'Stochastic_Gradient_Descent_AdaGrad','y-',i)
    y,i=Stochastic_Gradient_Descent_RMSProp(X,Y,maxtime=1000)[1],Stochastic_Gradient_Descent_RMSProp(X,Y,maxtime=1000)[3]
    plot_loss(f,y,'Stochastic_Gradient_Descent_RMSProp','c-',i)
    y,i=Adam(X,Y,maxtime=1000)[1],Adam(X,Y,maxtime=1000)[3]
    plot_loss(f,y,'Adam','k-',i)    
    y,i=AdaDelta(X,Y,maxtime=1000)[1],AdaDelta(X,Y,maxtime=1000)[3]
    plot_loss(f,y,'AdaDelta','m-',i)

    
    
output(300)  

