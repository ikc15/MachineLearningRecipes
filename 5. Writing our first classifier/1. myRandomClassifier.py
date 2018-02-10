# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 22:25:43 2018

@author: ikc15

Tutorial 5: Writing our own classifier.

Comment out 'choose classifier' code from tutorial 4: Let's write a pipeline'. 
We will write our own. 
"""
# In[1]:

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
# choose a classifer
'''
First we will just use a random classifier -> return a label randomly-> 
get right label 1/3 of the time (accuracy = 33%) because 3 possible labels. 
'''
import random 
class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return
    
    def predict(self, X_test):
        predictions = []
        for row in X_test: # each row of X_test corresponds to features of a label
            label = random.choice(self.y_train)
            predictions.append(label)
        return predictions
#from sklearn import tree # tree classifier
#clf = tree.DecisionTreeClassifier()
#
#from sklearn.neighbors import KNeighborsClassifier # K-nearest neighbours classifer
clf = ScrappyKNN()
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
