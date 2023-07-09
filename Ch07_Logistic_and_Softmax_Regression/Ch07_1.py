# Problem 1 of Chapter 7 "Logistic and Softmax Regression"
'''
Requires the package scikit-learn for plotting the confusion matrix
For GD & NR each generates 2 figures, confusion matrix and classified scatter
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sigmoid = lambda x:1/(1+np.exp(-x))

def Logistic_GD(
        x,      # N*d input matrix
        y,      # N*1 label vector
        delta = 0.1   # step size
):
    N = x.shape[0]
    d = x.shape[1]

    X = np.hstack((np.ones((N,1)),x)) # augmented N*(d+1) matrix of x
    w = np.random.rand(d+1,1)   #initialize w
    s = sigmoid(X@w)
    g = X.T@(s-y)   # gradient of log posterior
    
    tol = 1e-6  # tolerance
    n_iter = 0  # number of iterations

    while np.linalg.norm(g,ord=2) > tol:  # GD iteration till g is small
        w = w-delta*g   # update w
        s = sigmoid(X@w)
        g = X.T@(s-y)   # update g
        n_iter += 1

    print("Number of gradient descent iterations is %i"%n_iter)

    # assess the model performance
    y_pred = sigmoid(X@w)>0.5
    y_true = y.astype(int)
    accuracy = np.sum(y_pred == y_true)/N
    print("Accuracy of gradient descent is %.4f"%accuracy)

    cm = confusion_matrix(y_true,y_pred)    # generate confusion matrix
    print("Confusion matrix is:\n",cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion matrix by GD regression")
    plt.show()

    fig, ax = plt.subplots()
    for i in range(N):  # plot the data points
        if y_true[i,0] == 1:
            point_1 = ax.scatter(x[i,0],x[i,1],s=10,color='red',label="class 1")
        if y_true[i,0] == 0:
            point_0 = ax.scatter(x[i,0],x[i,1],s=10,color='blue',label="class -1")
    # plot the dividing line
    x_min = np.min(x[:,0])
    x_max = np.max(x[:,0])
    x1_grid = np.linspace(x_min,x_max,100)
    x2_grid = -(x1_grid*w[1,0]+w[0,0])/w[2,0]
    ax.plot(x1_grid,x2_grid,color='black')
    ax.legend(handles=[point_1,point_0])
    ax.set_title("GD logistic regression")
    fig.show()


def Logistic_NR(
        x,      # N*d input matrix
        y,      # N*1 label vector
        delta = 0.1   # step size
):
    N = x.shape[0]
    d = x.shape[1]

    X = np.hstack((np.ones((N,1)),x)) # augmented N*(d+1) matrix of x
    w = np.random.rand(d+1,1)   #initialize w
    s = sigmoid(X@w)
    g = X.T@(s-y)   # gradient of log posterior
    H = X.T@(np.diag(np.diag(s@(1-s).T)))@X # Hessian of log posterior
    
    tol = 1e-6  # tolerance
    n_iter = 0  # number of iterations

    while np.linalg.norm(g,ord=2) > tol:  # GD iteration till g is small
        w = w-delta*np.linalg.inv(H)@g   # update w
        s = sigmoid(X@w)
        g = X.T@(s-y)   # update gradient
        H = X.T@(np.diag(np.diag(s@(1-s).T)))@X # update Hessian
        n_iter += 1

    print("Number of gradient descent iterations is %i"%n_iter)

    # assess the model performance
    y_pred = sigmoid(X@w)>0.5
    y_true = y.astype(int)
    accuracy = np.sum(y_pred == y_true)/N
    print("Accuracy of Newton-Raphson is %.4f"%accuracy)

    cm = confusion_matrix(y_true,y_pred)    # generate confusion matrix
    print("Confusion matrix is:\n",cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion matrix by NR regression")
    plt.show()

    fig, ax = plt.subplots()
    for i in range(N):  # plot the data points
        if y_true[i,0] == 1:
            point_1 = ax.scatter(x[i,0],x[i,1],s=10,color='red',label="class 1")
        if y_true[i,0] == 0:
            point_0 = ax.scatter(x[i,0],x[i,1],s=10,color='blue',label="class -1")
    # plot the dividing line
    x_min = np.min(x[:,0])
    x_max = np.max(x[:,0])
    x1_grid = np.linspace(x_min,x_max,100)
    x2_grid = -(x1_grid*w[1,0]+w[0,0])/w[2,0]
    ax.plot(x1_grid,x2_grid,color='black')
    ax.legend(handles=[point_1,point_0])
    ax.set_title("NR logistic regression")
    plt.show()

data = np.loadtxt("./2ClassData.txt")
x = data[:,:2]
y = data[:,2].reshape(-1,1).astype(int) == 1

Logistic_GD(x,y)
Logistic_NR(x,y)

