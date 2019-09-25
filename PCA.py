# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 16:39:13 2019

@author: louis
"""
from Init import *

from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
from scipy.linalg import svd
import matplotlib.pyplot as plt

# TODO rename variable, Y is very confusing in this context. X_tilde maybe? (Felix)
Y = X - np.ones((N,1))*X.mean(0) # axis=0 returns means of all columns
Y1 = Y*(1/np.std(Y, 0))

U,S,Vh = svd(Y1,full_matrices=False)

V = Vh.T # transpose

Z = Y1 @ V # project Y1 on V (@ is a matrix multiplication)
# TODO try our hypothesis for 2.1.5. with this projection (higher influence ones with high value
 # make more of a posiitive influence on the projection)

i = 0
j = 5

f = figure()
clustering_info = 'scatterplot ' + name
title('river data: PCA ' + clustering_info)

# C made in init, length of class names. (eg 4 for seasons)
for c in range(C):
    class_mask = y == c  # y is list of season values corresponding to current observations
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

plt.savefig('plot.jpg')

# show()