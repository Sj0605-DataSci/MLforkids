#!/usr/bin/env python
# coding: utf-8

# # Support Vector Machines (SVM)
# 
# Support Vector Machine (SVM) is a powerful machine learning algorithm used for classification and regression tasks. The goal of SVM is to find the optimal hyperplane that separates the data into two classes. The hyperplane that SVM finds is the one that maximizes the margin between the closest points from each class. The points that lie on the margin are called support vectors.
# 
# SVM uses the Hinge Loss function to optimize the hyperplane. The Hinge Loss function is defined as:
# 
# $L(y_i, f(x_i)) = \max(0, 1 - y_i f(x_i))$
# 
# where $y_i$ is the true class label of the $i$th data point, $f(x_i)$ is the predicted score (dot product of feature vector and weight vector plus bias term) for the $i$th data point, and $\max(0, .)$ is the element-wise maximum function that returns 0 if the argument is positive and the argument itself if it is negative.
# 
# SVM also uses L2 regularization to control overfitting. The L2 regularization term adds a penalty to the loss function that is proportional to the square of the magnitude of the weight vector. The regularization parameter $\lambda$ controls the strength of the regularization.
# 
# The objective function of SVM with Hinge Loss and L2 regularization is:
# 
# $J(w, b) = \frac{1}{n} \sum_{i=1}^{n} \max(0, 1 - y_i (w^T x_i + b)) + \lambda |w|^2$
# 
# where $n$ is the number of training data points, $x_i$ is the feature vector of the $i$th data point, $y_i$ is its true class label, $w$ is the weight vector, $b$ is the bias term, $|w|^2$ is the L2 norm of the weight vector, and $\lambda$ is the regularization parameter.
# 
# The optimization problem of SVM is to find the values of $w$ and $b$ that minimize the objective function. The optimization can be performed using gradient descent or other optimization techniques.
# 
# The gradient of the objective function with respect to the weight vector and the bias term is:
# 
# $\nabla_w J(w, b) = \begin{cases} -\frac{1}{n} \sum_{i=1}^{n} y_i x_i & y_i (w^T x_i + b) < 1 \ -\frac{1}{n} \sum_{i=1}^{n} y_i x_i + 2\lambda w & otherwise \end{cases}$
# 
# $\nabla_b J(w, b) = \begin{cases} -\frac{1}{n} \sum_{i=1}^{n} y_i & y_i (w^T x_i + b) < 1 \ 0 & otherwise \end{cases}$

# In[2]:


import numpy as np


# In[3]:


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)


# In[ ]:


#Testing Time!!


# In[5]:


from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
y = np.where(y == 0, -1, 1)

clf = SVM()
clf.fit(X, y)
predictions = clf.predict(X)

print(clf.w, clf.b)


# In[ ]:


#Plotting Time!


# In[6]:


def visualize_svm():
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
    x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

    x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
    x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

    x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
    x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    ax.set_ylim([x1_min - 3, x1_max + 3])

    plt.show()


# In[7]:


visualize_svm()


# In[ ]:




