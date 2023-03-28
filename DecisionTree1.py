#!/usr/bin/env python
# coding: utf-8

# # Decision Tree
# 
# A decision tree is a tree-like model where each internal node represents a test on a feature, each branch represents the outcome of the test, and each leaf node represents a class label. The decision tree is constructed by recursively partitioning the data into subsets based on the features and their values.
# 
# The decision tree algorithm uses a metric called Information Gain to decide which feature to split on at each node. Information Gain measures the expected reduction in entropy (or increase in information) caused by splitting the data on a particular feature. The feature that results in the highest Information Gain is selected as the split feature.
# 
# Entropy is a measure of the impurity or randomness of a set of examples. Entropy is defined as follows:
# 
# \begin{equation}
# H(S) = - \sum_{i=1}^{c} p_i \log_2(p_i)
# \end{equation}
# 
# where $S$ is the set of examples, $c$ is the number of classes, $p_i$ is the proportion of examples in class $i$, and $\log_2$ is the base-2 logarithm.
# 
# Information Gain is defined as the difference between the entropy of the parent node and the weighted average of the entropy of the child nodes:
# 
# \begin{equation}
# IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)
# \end{equation}
# 
# where $S$ is the set of examples at the parent node, $A$ is the feature being considered for splitting, $Values(A)$ is the set of possible values for the feature $A$, $|S_v|$ is the number of examples in $S$ that have the value $v$ for feature $A$, and $|S|$ is the total number of examples at the parent node.
# 
# The decision tree algorithm selects the feature with the highest Information Gain as the split feature at each node, and recursively splits the data based on the values of that feature. The algorithm stops splitting when all examples at a node belong to the same class or when no further splits can be made.
# 
# The decision tree model can be represented as a tree structure, where each internal node represents a split on a feature, and each leaf node represents a class label. The decision tree can be used for prediction by traversing the tree from the root node to a leaf node based on the feature values of the example being classified.
# 
# The decision tree algorithm has several variants and extensions, such as the ID3, C4.5, and CART algorithms, which differ in their splitting criteria and handling of categorical and continuous features.

# In[1]:


import numpy as np
from collections import Counter


# In[2]:


def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


# In[3]:


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


# In[4]:


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if ( depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # greedily select the best split according to information gain
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)

        # grow the children that result from the split
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        # parent loss
        parent_entropy = entropy(y)

        # generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # compute the weighted avg. of the loss for the children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # information gain is difference in loss before vs. after split
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common


# In[ ]:


#Testing Time!!


# In[5]:


from sklearn import datasets
from sklearn.model_selection import train_test_split

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


# In[6]:


bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


# In[7]:


clf = DecisionTree(max_depth=10)
clf.fit(X_train, y_train)


# In[8]:


y_pred = clf.predict(X_test)


# In[9]:


acc = accuracy(y_test, y_pred)
print("Accuracy:", acc)


# In[ ]:





# In[5]:





# In[ ]:





# In[ ]:





# In[ ]:




