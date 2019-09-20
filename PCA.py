# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 16:39:13 2019

@author: louis
"""
from Init import *

from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
from scipy.linalg import svd

Y = X - np.ones((N,1))*X.mean(0)
Y1 = Y*(1/np.std(Y,0))

U,S,Vh = svd(Y1,full_matrices=False)

V = Vh.T    

Z = Y1 @ V

i = 0
j = 1

f = figure()
title('river data: PCA')

for c in range(C):
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

show()