# Problem 3 of Chapter 5 "Linear Regression"
# This code includes the 2 subsections. 

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

        print(">>> Plot the 1D regression curve")
        NUM = 100
        x_grid = np.linspace(np.min(x)-1,np.max(x)+1,NUM).reshape(-1,1)
        fig1 = plt.figure(1).add_subplot(111)
        fig1.scatter(x,y,facecolors='none',edgecolors='blue',label=r"$y$") # data points
        fig1.plot(x_grid,np.matmul(np.hstack((np.ones((NUM,1)),x_grid)),w),
                  label=r"$\hat{y}$") # regression curve
        fig1.plot(x_grid,y_bar*np.ones((NUM,1)),
                  label=r"$\bar{y}$",
                  linestyle='dashed') # average
        fig1.legend()
        fig1.set_title("Ch05_3a")
        
    if d==2:
        print(">>> Plot the 2D regression curve")
        NUM = 100
        x1 = x[:,0]
        x2 = x[:,1]
        x_grid = np.linspace(np.min(x1)-1,np.max(x1)+1,NUM).reshape(-1,1)
        y_grid = np.linspace(np.min(x2)-1,np.max(x2)+1,NUM).reshape(-1,1)
        x_grid, y_grid = np.meshgrid(x_grid,y_grid)

        z = np.empty((NUM, NUM))
        for i in range(NUM):
            for j in range(NUM):
                z[i, j] = w[0,0]+w[1,0]*x_grid[i,j]+w[2,0]*y_grid[i,j]
        fig2 = plt.figure(2).add_subplot(projection='3d')
        fig2.scatter(x1,x2,y,
                     facecolors='none',
                     edgecolors='blue',
                     lw=1,
                     label=r"$y$") # data points
        fig2.plot_surface(x_grid,y_grid,z,
                          cmap='viridis',
                          alpha=0.8)
        fig2.legend()
        fig2.set_xlabel(r'$x_1$')
        fig2.set_ylabel(r'$x_2$')
        fig2.set_title("Ch05_3b")
        plt.show()


data1 = np.loadtxt("dataLS1.txt")
# first need to convert x,y from 1D to 2D
LS_Regression(x=data1[:,0].reshape(-1,1),y=data1[:,1].reshape(-1,1))

data2 = np.loadtxt("dataLS2.txt")
LS_Regression(x=data2[:,:2],y=data2[:,2].reshape(-1,1))
