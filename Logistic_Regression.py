#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression Equation
# The logistic regression model is given by:
# 
# $$\log \frac{p}{1-p} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_p x_p$$
# 
# where $p$ is the probability of the outcome variable taking on the value 1, $x_1$ through $x_p$ are the predictor variables, and $\beta_0$ through $\beta_p$ are the coefficients.
# 
# The predicted probability of the outcome variable taking on the value 1 is given by:
# 
# $$p = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_p x_p)}}$$

# In[31]:


import numpy as np
import warnings


# In[ ]:


warnings.filterwarnings('ignore')


# In[24]:


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

class LoR: #Logistic Regression
    
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
            linear_model = np.dot(X,self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        
    def predict(self,X):
        linear_model = np.dot(X,self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)
            
    def sigmoid(self,x):
        x = np.float64(x)
        return 1 / (1 + np.exp(-x))


# In[25]:


#Testing Time!


# In[26]:


from sklearn import datasets
from sklearn.model_selection import train_test_split


# In[27]:


breast = datasets.load_breast_cancer()
X,y = breast.data, breast.target


# In[28]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1234)


# In[34]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[35]:


y_train


# In[33]:


cl = LoR(0.001,1000)
cl.fit(X_train,y_train)
predictions = cl.predict(X_test)
print(accuracy(y_test,predictions))


# In[ ]:




