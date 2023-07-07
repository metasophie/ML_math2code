# Problem 1 of Chapter 7 "Logistic and Softmax Regression"
'''
requires the package scikit-learn for plotting the confusion matrix
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

    while np.linalg.norm(g,ord=2)>tol:  # GD iteration till g is small
        w = w-delta*g   # update w
        s = sigmoid(X@w)
        g = X.T@(s-y)   # update g
        n_iter += 1

    print("Number of gradient descent iterations is %i"%n_iter)

    # assess the model performance
    y_pred = sigmoid(X@w)>0.5
    y_true = y.astype(int)
    accuracy = np.sum(y_pred == y_true)/float(N)
    print("Accuracy of gradient descent is %.4f"%accuracy)

    cm = confusion_matrix(y_true,y_pred)    # generate confusion matrix
    print("Confusion matrix is:\n",cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion matrix by GD regression")
    plt.show()

def Logistic_NR(
        x,      # N*d input matrix
        y,      # N*1 label vector
)

data = np.loadtxt("./2ClassData.txt")
x = data[:,:2]
y = data[:,2].reshape(-1,1).astype(int) == 1

Logistic_GD(x,y)

