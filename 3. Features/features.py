# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 13:29:37 2018

@author: ikc15
"""
# 1. Features 
#%%%%%
'''
 Need to pick good features to collect data from such that they can distinguish 
 between the target labels.
 E.g. Height is a good feature for distinguishing two breeds of dogs- but not 
 perfect as seen by overlap region in histogram -> these regions are ambiguous.
 However, eye color would not be a good feature as it is independent of 
 the breed. 
'''
#%%%%%
import matplotlib.pyplot as plt
import numpy as np

greyhound, labrador = 500, 500

grey_height = 28 + 4*np.random.randn(greyhound)
lab_height = 24 + 4*np.random.randn(labrador)

plt.hist([grey_height, lab_height], stacked = True, color = ['r', 'b'], label=['grey', 'lab'])
plt.legend()
plt.show()

