'''
Problem of Chapter 9 "Feature Extraction"
'''

import numpy as np
import matplotlib.pyplot as plt

# load the data
iris = np.loadtxt("iris.txt")
X = iris[:,:4]
y = iris[:,4].astype(int)
N = y.shape[0]
K = 3

# visualize the first 2 dimensions of the dataset based on class
fig = plt.figure(1)
for i in range(1,K+1):
    plt.scatter(X[y==i,0], X[y==i,1], s=20, label='class %i'%i)

fig.legend()
plt.show()

# calculate distances of class centers
n = np.zeros(K)
mean = np.zeros((K,4))
cov = np.zeros((K,4,4))
for i in range(1,K+1):  # compute mean & covariance matrix for each class
    n[i-1] = np.sum(y==i)
    mean[i-1,:] = np.average(X[y==i,:],axis=0)
    cov[i-1,:,:] = np.cov(X[y==i,:],rowvar=False)*(n[i-1]-1)/n[i-1]
'''
Note that the numpy function cov differs from Sigma_i in the book, 
thus the factor at the end
'''

d1 = np.zeros((K,K))    # norm = 1
d2 = np.zeros((K,K))    # norm = 2
dinf = np.zeros((K,K))  # norm = inf
dB = np.zeros((K,K))    # Bhattacharyya distance

for i in range(1,K+1):
    for j in range(1,i):
        d = mean[i-1,:]-mean[j-1,:]
        d1[i-1,j-1] = np.linalg.norm(d,ord=1)
        d2[i-1,j-1] = np.linalg.norm(d,ord=2)
        dinf[i-1,j-1] = np.linalg.norm(d,ord=np.inf)
        dB[i-1,j-1] = d@np.linalg.inv((cov[i-1,:,:]+cov[j-1,:,:])/2)@d.T/4+np.log(np.linalg.det((cov[i-1,:,:]+cov[j-1,:,:])/2)/np.sqrt(np.linalg.det(cov[i-1,:,:])*np.linalg.det(cov[j-1,:,:])))

np.set_printoptions(precision=4)
print("d1 = \n",d1)
print("d2 = \n",d2)
print("dinf = \n",dinf)
print("dB = \n",dB)

# calculate scatter matrices
total_mean = np.average(mean,axis=0,weights=n)

SB = np.zeros((4,4))
for i in range(1,K+1):
    SB += (mean[i-1,:]-total_mean).reshape(-1,1)@(mean[i-1,:]-total_mean).reshape(1,-1)*n[i-1]/N
    
SW = np.average(cov,axis=0,weights=n)
ST = np.cov(X,rowvar=False)*(N-1)/N

print(SB)
print(SW)
print(ST)

print(np.sum(ST-SW-SB))