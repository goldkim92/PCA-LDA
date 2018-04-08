# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 23:27:24 2018

@author: USER
"""
#%% import
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets

#%% extract data
iris = datasets.load_iris()
X = iris.data
X = X.T
y = iris.target
target_names = iris.target_names

#%% PCA
# mean
m = np.mean(X,axis=1, keepdims=True)

# scatter matrix
S = np.matmul(X-m, (X-m).T) 

# eigenvalue & eigenvector
e_value, e_vector = np.linalg.eig(S)

# two eigenvectors corresponding maximum two eigenvalues
e = e_vector[:,0:2]

# PCA points
a = np.matmul(e.T, X-m) 

# figure
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

plt.figure()

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(a[0,y == i], a[1,y == i], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')


#%% LDA
X_0 = X[:,y==0]
X_1 = X[:,y==1]
X_2 = X[:,y==2]
m_0 = np.mean(X_0, axis=1, keepdims=True)
m_1 = np.mean(X_1, axis=1, keepdims=True)
m_2 = np.mean(X_2, axis=1, keepdims=True)

# within-class scatter matrix
S_w = sum([np.matmul(xx-mm,(xx-mm).T) for xx, mm in zip([X_0,X_1,X_2],[m_0,m_1,m_2])])

# between-class scatter matrix
S_b = sum([nn * np.matmul(m-mm,(m-mm).T) for nn, mm in \
           zip([X_0.shape[1],X_0.shape[1],X_0.shape[1]],[m_0,m_1,m_2])])

# eigenvalue & eigenvector
SS = np.matmul(np.linalg.pinv(S_w), S_b)
e_value, e_vector = np.linalg.eig(SS)

# two eigenvectors corresponding maximum two eigenvalues
e = e_vector[:,0:2]

# LDA points
a = np.matmul(e.T, X) 

# figure
plt.figure()

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(a[0,y == i], a[1,y == i], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')
