# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 01:03:39 2018

@author: ikc15

Train an Image Classifier with TensorFlow Learn to classify handwritten digits 
from the mnist dataset.

Since we are using a neural network. We do not need to manually select features. 
The neural network takes each raw pixel as a feature (ie. 784 features in this 
example) 
"""
# In[]:
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
learn = tf.contrib.learn
tf.logging.set_verbosity(tf.logging.ERROR)

# In[]:
'Import the dataset'
mnist = learn.datasets.load_dataset('mnist')
data = mnist.train.images
labels = np.asarray(mnist.train.labels, dtype=np.int32)
test_data = mnist.test.images
test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

# In[]:
'Display digits'
def display(i):
    img = test_data[i]
    plt.title('Example %d, Label: %d' %(i, test_labels[i]))  
    # reshape to 28x28 pixels because the image was flattened to a 1darray of length 784
    plt.imshow(img.reshape((28,28)), cmap=plt.cm.gray_r) 
    
# In[]:
'''
Fit a linear classifier (neural netowrk with input layer (794 nodes) -> output layer (10 nodes))-
Our goal here is to get about 90% accuracy with this simple classifier. 
'''
feature_columns = learn.infer_real_valued_columns_from_input(data)
# Choose classifier
#1st arg: 10 classes- one for each digit 0-9, 2nd arg: informs the classifier about the input data
clf = learn.LinearClassifier(n_classes=10, feature_columns=feature_columns) 
# train classifier (using gradient decent) to adjust weights of each input-output connection for all training data
clf.fit(data,labels, batch_size=100, steps=1000) 

# In[]:
'Evaulate accuracy of classifier'
accuracy = clf.evaluate(test_data, test_labels)
print (accuracy['accuracy'])

# In[]:
'Classify a few examples'
pred = clf.predict(np.array([test_data[0]]), as_iterable=False)
print("Predicted %d, Label: %d" % (pred, test_labels[0]))
display(0)
#print ('Predicted: %d, label: %d'%(clf.predict(test_data[0]), test_labels[0]))

# In[]:
'Visualize learned weights'
f, ax = plt.subplots(2,5, figsize=(10,4))
ax = ax.reshape(-1)
#for var in clf.get_variable_names():
#    print("var:", var, "=", clf.get_variable_value(var))
weights = clf.get_variable_value(clf.get_variable_names()[1])
for i in range(len(ax)):
    a = ax[i]
    a.imshow(weights.T[i].reshape(28,28), cmap=plt.cm.seismic)
    a.set_title(i)
    a.set_xticks(()) # ticks be gone
    a.set_yticks(())
plt.show()
f.savefig('weights.png', bbox_inches='tight')