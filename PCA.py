#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # covariance, function needs samples as columns
        cov = np.cov(X.T)

        # eigenvalues, eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # -> eigenvector v = [:,i] column vector, transpose for easier calculations
        # sort eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # store first n eigenvectors
        self.components = eigenvectors[0 : self.n_components]

    def transform(self, X):
        # project data
        X = X - self.mean
        return np.dot(X, self.components.T)


# In[3]:


import matplotlib.pyplot as plt
from sklearn import datasets


# In[4]:


# data = datasets.load_digits()
data = datasets.load_iris()
X = data.data
y = data.target


# In[5]:


# Project the data onto the 2 primary principal components
pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)


# In[6]:


print("Shape of X:", X.shape)
print("Shape of transformed X:", X_projected.shape)


# In[10]:


x1 = X_projected[:, 0]
x2 = X_projected[:, 1]


# In[11]:


plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar
plt.scatter(x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3))
plt.show()


# In[ ]:





# In[ ]:




