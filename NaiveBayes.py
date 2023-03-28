#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes
# 
# The Naive Bayes algorithm is a supervised learning method used for classification tasks. It is based on Bayes' theorem and the assumption of conditional independence between the features given the class label.
# 
# The Naive Bayes algorithm can be summarized as follows:
# 
# 1. Learn the prior probabilities of each class label based on the frequency of occurrence in the training set.
# 2. Learn the conditional probabilities of each feature given each class label based on the frequency of occurrence in the training set.
# 3. For a given test data point, calculate the posterior probability of each class label using Bayes' theorem and the learned probabilities.
# 4. Assign the class label with the highest posterior probability to the test data point.
# 
# In the case of continuous data, the conditional probabilities of each feature given each class label can be modeled using a Gaussian distribution. The formula for the conditional probability of feature `X` given class label `Ck` can be written as:
# 
# $$P(X_i|C_k) = \frac{1}{\sqrt{2\pi\sigma_{k,i}^2}} \exp\left(-\frac{(X_i - \mu_{k,i})^2}{2\sigma_{k,i}^2}\right)$$
# 
# where `X_i` represents the `i`th feature of the test data point, `Ck` represents the `k`th class label, `mu_k,i` represents the mean of the `i`th feature for class label `Ck`, and `sigma_k,i` represents the standard deviation of the `i`th feature for class label `Ck`.
# 
# The prior probability of each class label `P(Ck)` can be calculated as the frequency of occurrence of class label `Ck` in the training set. The posterior probability of each class label for a given test data point can be calculated using Bayes' theorem:
# 
# $$P(C_k|X) = \frac{P(C_k) \prod_{i=1}^{n} P(X_i|C_k)}{P(X)}$$
# 
# where `n` is the number of features, `P(X_i|C_k)` is the conditional probability of the `i`th feature given class label `Ck`, and `P(X)` is a normalization factor that ensures that the sum of the posterior probabilities over all classes is equal to one.
# 
# The class label with the highest posterior probability is assigned to the test data point.

# In[2]:


import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn import datasets


# In[3]:


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy 

class NaiveBayes:
        def fit(self, X, y):
            n_samples, n_features = X.shape
            self._classes = np.unique(y)
            n_classes = len(self._classes)
            
            # calculate mean, var, and prior for each class
            self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
            self._var = np.zeros((n_classes, n_features), dtype=np.float64)
            self._priors = np.zeros(n_classes, dtype=np.float64)

            for idx, c in enumerate(self._classes):
                X_c = X[y == c]
                self._mean[idx, :] = X_c.mean(axis=0)
                self._var[idx, :] = X_c.var(axis=0)
                self._priors[idx] = X_c.shape[0] / float(n_samples)

        def predict(self, X):
            y_pred = [self._predict(x) for x in X]
            return np.array(y_pred)

        def _predict(self, x):
            posteriors = []

        # calculate posterior probability for each class
            for idx, c in enumerate(self._classes):
                prior = np.log(self._priors[idx])
                posterior = np.sum(np.log(self._pdf(idx, x)))
                posterior = prior + posterior
                posteriors.append(posterior)

        # return class with highest posterior probability
            return self._classes[np.argmax(posteriors)]

        def _pdf(self, class_idx, x):
            mean = self._mean[class_idx]
            var = self._var[class_idx]
            numerator = np.exp(-((x - mean) ** 2) / (2 * var))
            denominator = np.sqrt(2 * np.pi * var)
            return numerator / denominator


# In[4]:


#Testing Time!!


# In[5]:


X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# In[6]:


nb = NaiveBayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)


# In[7]:


print("Naive Bayes classification accuracy", accuracy(y_test, predictions))

