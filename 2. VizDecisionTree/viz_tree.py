# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 23:16:50 2018

@author: ikc15
"""
import numpy as np
from sklearn import tree

# In[]:
# 1. Import dataset
from sklearn.datasets import load_iris
###%

iris = load_iris()
test_idx = [0, 50, 100] # location of the first of each type of flower
#print (iris.feature_names)
#print (iris.target_names)
#for i in range(len(iris.target)):
#   print('Exmaple %d: label %s, features %s'%(i, iris.target[i], iris.data[i]))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 2. Train classifier
# Split dataset into training and testing data. 

# training data
train_target = np.delete(iris.target, test_idx)  # remove these targets from dataset for testing later
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# classifier - decision tree
clf = tree.DecisionTreeClassifier()
#train clf on traning data
clf.fit(train_data, train_target)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 3. Predict label for test data (unseen by clf)
print (test_target) # what we should get
print (clf.predict(test_data)) # prediction 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 4. Vizualise code

from sklearn.externals.six import StringIO
import pydotplus
dot_data = StringIO()
tree.export_graphviz(clf,
                         out_file=dot_data,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         impurity=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('iris.pdf')