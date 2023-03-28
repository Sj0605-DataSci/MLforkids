#!/usr/bin/env python
# coding: utf-8

# # K-Nearest Neighbors (KNN)
# 
# The k-nearest neighbors algorithm is a supervised learning method used for classification and regression tasks. The basic idea is to classify or predict the target variable based on the k nearest data points in the training set.
# 
# For a given test data point, the KNN algorithm identifies the k nearest neighbors in the training set based on a distance metric (e.g., Euclidean distance or Manhattan distance) and assigns the most common class (in the case of classification) or the average value (in the case of regression) of those neighbors to the test data point.
# 
# The KNN algorithm can be summarized as follows:
# 
# 1. Choose the value of `k` (the number of nearest neighbors to consider).
# 2. For each test data point, calculate the distance to all data points in the training set.
# 3. Select the `k` data points in the training set with the smallest distances to the test data point.
# 4. Assign the most common class (in the case of classification) or the average value (in the case of regression) of those neighbors to the test data point.
# 
# The equation for KNN can be written as follows:
# 
# $$y = \operatorname{arg\,max}_{c_j} \sum_{i=1}^{k} w(i) [y_{j} = c_i]$$
# 
# where `y` is the predicted class or value for the test data point, `c` represents the different classes or values, `w(i)` is the weight assigned to the `i`th nearest neighbor (e.g., the inverse of its distance to the test data point), and `yj` is the class or value of the `j`th nearest neighbor.
# 
# 

# In[40]:


import numpy as np
from collections import Counter


# In[41]:


def euclidian(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))       # Euclidian Distance 

class KNN:                                   #K-nearest Neighbour
    
    def __init__(self,k=3):                  #taking a default value of 3
        self.k = k
    
    def fit(self,X,y):                      #just like when we use scikit learn we define a function 'fit', similar to it
        self.X_train = X
        self.y_train = y
        
    def predict(self,X):                   #for predicting labels
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
    
    def _predict(self,x):                 # maths behind predicting labels
        
        #compute the distances of new sample with training samples
        distance = [euclidian(x,x_train) for x_train in X_train]
        
        #lets get indices
        K_indices = np.argsort(distance)[:self.k]
        
        #lets get labels
        K_nearest_labels = [self.y_train[i] for i in K_indices]
        
        #common label
        most_common = Counter(K_nearest_labels).most_common(1)
        return most_common[0][0]
        


# In[42]:


# Testing time !


# In[43]:


from sklearn import datasets
from sklearn.model_selection import train_test_split


# In[44]:


iris = datasets.load_iris()
X,y = iris.data, iris.target


# In[45]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1234)


# In[46]:


clf = KNN(k=5)


# In[47]:


clf.fit(X_train,y_train)


# In[48]:


predictions = clf.predict(X_test)


# In[49]:


accuracy = (np.sum(predictions == y_test))/len(y_test)


# In[50]:


print(accuracy)


# In[51]:


#Plotting


# In[52]:


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])


# In[53]:


plt.figure()
plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap,edgecolor='k',s=20)
plt.show()


# In[ ]:




