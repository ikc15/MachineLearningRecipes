# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 13:45:36 2018

@author: ikc15
"""
# In[1]:

'''
A classifier simply does 'logic' on features to determine label -> i.e. f(X) = y. 
'''

# Import a dataset
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
y = iris.target

# In[2]:
# Partition data for training and testing

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .5) # use half of the data for testing

# In[3]:
# choose classifier

from sklearn import tree # tree classifier
clf = tree.DecisionTreeClassifier()

from sklearn.neighbors import KNeighborsClassifier # K-nearest neighbours classifer
clf = KNeighborsClassifier()

# In[4]:
# train classifier
clf.fit(X_train, y_train)

# test classifier
pred = clf.predict(X_test)

# Compute accuracy
counter = 0
for i in range (len(pred)):
    if (pred[i]-y_test[i]==0): continue
    else: 
        counter +=1
print ('Accuracy = %.2f%%' %((1.-counter/len(pred))*100))

# In[]:
'''
Learning = using training data to adjust the parameters of a model to increase the accuracy of prediction.
'''