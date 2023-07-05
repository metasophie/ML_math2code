# Problem 3 of Chapter 5 "Linear Regression"
# Yuanzhe Dong 2023

import numpy as np
import matplotlib.pyplot as plt

# input
def LS_Regression(
    x, # N*d matrix
    y  # N*1 vector 
):
    
    N = x.shape[0]
    d = x.shape[1]

    X = np.hstack((np.ones((N,1)),x)) # augmented matrix of x
    PInvX = np.linalg.pinv(X)     # pseudo-inverse of X
    w = np.matmul(PInvX,y)

    y_bar = np.average(y)
    TSS = np.sum(np.power(y-y_bar,2))
    y_hat = np.matmul(X,w)
    ESS = np.sum(np.power(y_hat-y_bar,2))
    RSS = TSS-ESS
    R_squared = ESS/TSS

    print(">>> Results of regression: \n w=")
    print(w)
    print(r"TSS=%.4f ESS=%.4f RSS=%.4f R2=%.4f"%(TSS,ESS,RSS,R_squared))
    if d==1:
        rho = np.corrcoef(x,y,rowvar=False)[0,1]
        print(r"In this special case where d=1, rho=%.4f"%rho)

        print(">>> Plot the regression curve")

data1 = np.loadtxt("dataLS1.txt")
LS_Regression(x=data1[:,0].reshape(-1,1),y=data1[:,1].reshape(-1,1))
# convert x,y from 1D to 2D

