# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 17:23:03 2019

@author: louis
"""
from Init import *
from PCA import *

import matplotlib.pyplot as plt
from scipy.linalg import svd

rho = (S*S) / (S*S).sum() # creates array with influences of each component

threshold = 0.9

plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()
