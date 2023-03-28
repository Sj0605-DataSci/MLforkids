#!/usr/bin/env python
# coding: utf-8

# # Linear Regression Equation
# 
# The equation for a linear regression model with `p` predictor variables is:
# 
# $$
# y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_p x_p + \epsilon
# $$
# 
# where `y` is the dependent variable, `x1` through `xp` are the predictor variables, `β0` is the intercept term, `β1` through `βp` are the coefficients, and `ϵ` is the error term.

# In[30]:


import numpy as np
import matplotlib.pyplot as plt


# In[31]:


class LiR: #Linear Regression
    
    def __init__(self,lr=0.001,n_iters=1000): #taking a default value of 3
        self.lr = lr
        self.n_iters=n_iters
        self.weights = None
        self.bias = None
    
    def fit(self,X,y): #just like when we use scikit learn we define a function 'fit', similar to it
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            
            #formula for y_prediction
            y_predicted = np.dot(X,self.weights) + self.bias
            
            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        
    def predict(self,X): #for predicting labels
        y_predicted = np.dot(X,self.weights) + self.bias
        return y_predicted
       


# In[ ]:


#Testing time !


# In[33]:


from sklearn.model_selection import train_test_split
from sklearn import datasets
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


# In[34]:


regressor = LiR(0.01,1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)


# In[35]:


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# In[36]:


mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)


# In[37]:


def r2_score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr ** 2


# In[38]:


accu = r2_score(y_test, predictions)
print("Accuracy:", accu)


# In[39]:


y_pred_line = regressor.predict(X)


# In[40]:


#Plotting


# In[41]:


cmap = plt.get_cmap("viridis")
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
plt.show()


# In[ ]:




