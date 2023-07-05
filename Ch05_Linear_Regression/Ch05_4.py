# Problem 4 of Chapter 5 "Linear Regression"
'''
First we generate x1 & y data with y=2*x1+noise
Then for noise level a=0.01, 0.1, 1, we generate 3 differently perturbed x2 based on x1
For each a & x2, we apply 3 ridge regression models with lambda=0, 0.01, 0.1
'''

import numpy as np
import matplotlib.pyplot as plt

def Ridge_Regression(
       x, # N*d matrix
       y, # N*1 matrix
       lam  # weight decay parameter
):
    N = x.shape[0]
    d = x.shape[1]

    X = np.hstack((np.ones((N,1)),x)) # augmented matrix of x
    Xinv = np.linalg.inv(X.T@X+lam*np.eye(d+1))@X.T  # pseudo-inverse modified by weight decay
    w = Xinv@y

    y_bar = np.average(y)
    TSS = np.sum(np.power(y-y_bar,2))
    y_hat = X@w
    ESS = np.sum(np.power(y_hat-y_bar,2))
    R_squared = ESS/TSS

    return (w,R_squared)

# generate data
x1 = np.array([2.4, 3.1, 3.8, 2.3, 2.0, 3.7, 3.2, 3.0, 2.8, 1.6]).reshape(-1,1)
y = np.array([4.6, 6.1, 7.7, 4.9, 4.1, 7.4, 6.3, 5.8, 5.5, 3.4]).reshape(-1,1)

for a in (0.01,0.1,1):

    fig = plt.figure(figsize=(10,10))
    
    plt.axis('off')
    plt.title("noise level a=%.2f"%a)
    
    for i in range(3):

        x2 = x1+a*np.random.rand(10,1) # generate x2

        # evaluate how ill-conditioned the problem is through rho and eigenvalues
        rho = np.corrcoef(x1,x2,rowvar=False)[0,1]
        X = np.hstack((np.ones((10,1)),x1,x2))
        eigenvals = np.linalg.eigvals(X.T@X)
        print("Under noise level a=%.2f, rho=%.5f, eigenvalues are"%(a,rho))
        print(eigenvals)

        for j in range(3):

            lam = (0,0.01,0.1)[j]
            ax = fig.add_subplot(3,3,3*j+i+1,projection='3d')
            w, R_squared = Ridge_Regression(np.hstack((x1,x2)),y,lam)

            NUM = 100
            x_grid = np.linspace(0,5,NUM)
            x_grid, y_grid = np.meshgrid(x_grid,x_grid)
            z = np.empty((NUM, NUM))
            for k in range(NUM):
                for l in range(NUM):
                    z[k, l] = w[0,0]+w[1,0]*x_grid[k,l]+w[2,0]*y_grid[k,l]
            ax.scatter(x1,x2,y,
                     facecolors='none',
                     edgecolors='blue',
                     lw=1) # data points
            ax.plot_surface(x_grid,y_grid,z,
                            cmap='viridis',
                            alpha=0.8)
            
            ax.set_title(r"$\lambda$=%.2f,$R^2$=%.4f"%(lam,R_squared)+"\n"+r"$\mathbf{w}$=(%.4f,%.4f,%.4f)"%(w[0,0],w[1,0],w[2,0]),
                         y=0.92,
                         fontsize='x-small') # display the parameters

    plt.show()
    
