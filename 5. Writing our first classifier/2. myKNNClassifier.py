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
# choose classifier
from scipy.spatial import distance 

def euc(a,b):
    'Computes distnace between pts a (in testing data) and b (in training data).'
    return distance.euclidean(a,b)

class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return
    
    def predict(self, X_test):
        predictions = []
        for row in X_test:              # each row of X_test corresponds to features of a label
            label = self.closest(row)   # find closest training data pt and its label and predict label of 'row' to be the same.
            predictions.append(label)
        return predictions 
    
    def closest(self, row):
        ''''
        finds closest training data pt to input test data pt. 
        Input: row = features of testing data
        Output: label = label of the closest training data pt. 
        '''
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1,len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if (dist<best_dist):
                best_dist = dist
                best_index = i
            label = self.y_train[best_index]
        return label
        
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
