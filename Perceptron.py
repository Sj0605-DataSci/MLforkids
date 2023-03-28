#!/usr/bin/env python
# coding: utf-8

# # Perceptron
# 
# The Perceptron algorithm is a supervised learning method used for binary classification tasks. It is a linear classification model that learns a decision boundary that separates the two classes.
# 
# The Perceptron algorithm can be summarized as follows:
# 
# 1. Initialize the weight vector `w` and the bias term `b` to zero or small random values.
# 2. For each training data point, compute the weighted sum of the feature vector `x` and the weight vector `w`, and add the bias term `b`.
# 3. If the result is positive, predict the positive class label. If the result is negative, predict the negative class label.
# 4. Update the weight vector and the bias term if the prediction is incorrect, using the following update rule:
# 
#    $$w \leftarrow w + \alpha y x$$
#    $$b \leftarrow b + \alpha y$$
# 
#    where `alpha` is the learning rate, `y` is the true class label (either +1 or -1), and `x` is the feature vector of the training data point.
# 
# 5. Repeat steps 2-4 for a fixed number of epochs or until convergence.
# 
# The decision boundary learned by the Perceptron algorithm is a hyperplane defined by the equation:
# 
# $$w^T x + b = 0$$
# 
# where `w` is the weight vector, `x` is the feature vector, and `b` is the bias term.
# 
# The Perceptron algorithm updates the weight vector and the bias term in order to minimize the total classification error on the training data set. This can be expressed as the following loss function:
# 
# $$L(w,b) = -\sum_{i=1}^{N} y_i(w^T x_i + b)$$
# 
# where `N` is the number of training data points, `yi` is the true class label of the `i`th training data point, `xi` is the feature vector of the `i`th training data point, and the sum is taken over all training data points.
# 
# The Perceptron algorithm uses stochastic gradient descent to minimize the loss function. The update rule for stochastic gradient descent can be written as:
# 
# $$w \leftarrow w - \alpha \nabla_w L(w,b)$$
# $$b \leftarrow b - \alpha \nabla_b L(w,b)$$
# 
# where `alpha` is the learning rate, and `∇w L(w,b)` and `∇b L(w,b)` are the gradients of the loss function with respect to `w` and `b`, respectively.

# In[3]:


import numpy as np


# In[14]:


class Perceptron:
    
    def __init__(self,learning_rate=0.001,n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activationfunc = self.unitstepfunc
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iters):

            for idx, x_i in enumerate(X):

                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activationfunc(linear_output)

                # Perceptron update rule
                update = self.lr * (y_[idx] - y_predicted)

                self.weights += update * x_i
                self.bias += update
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activationfunc(linear_output)
        return y_predicted
    
    def unitstepfunc(self,x):
        return np.where(x>=0,1,0)


# In[15]:


#Testing Time!!


# In[16]:


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


# In[17]:


X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# In[18]:


p = Perceptron(learning_rate=0.01, n_iters=1000)
p.fit(X_train, y_train)
predictions = p.predict(X_test)


# In[19]:


print("Perceptron classification accuracy", accuracy(y_test, predictions))


# In[20]:


#Plotting Time!


# In[21]:


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

x0_1 = np.amin(X_train[:, 0])
x0_2 = np.amax(X_train[:, 0])

x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

ymin = np.amin(X_train[:, 1])
ymax = np.amax(X_train[:, 1])
ax.set_ylim([ymin - 3, ymax + 3])

plt.show()


# In[ ]:




