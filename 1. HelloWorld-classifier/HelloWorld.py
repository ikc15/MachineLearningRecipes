# -*- coding: utf-8 -*-
"""
Spyder Editor

ML1: Classifier - Decision tree
"""

from sklearn import tree


#Training Data
features = [[150, 0], [170, 0],[140, 1], [130, 1]] # 'bumpy' = 0, 'smooth'=1
labels = ['orange', 'orange','apple','apple']

# Create Classifier: Decision tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels) # 'fit' finds pattern in data
print (clf.predict([[160,1]]))