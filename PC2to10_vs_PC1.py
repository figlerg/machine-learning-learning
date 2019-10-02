from matplotlib.pyplot import (figure, subplot, plot, xlabel, ylabel, 
                               xticks, yticks,legend,show,close)


import matplotlib.pyplot as plt
import numpy as np
from Init import *
from PCA import *

#M=10
#

colors =['r','y','b']

M=9
figure()    #figsize=(12,10)
m1=0
for m2 in range(M):
    subplot(3,3, m2 + 1)
    for c in range(C):
        class_mask = (y==c)
        plot(np.array(Z[class_mask,m1]), np.array(Z[class_mask,m2+1]),colors[c]+'.',markersize=6, alpha=.8)
        ylabel('PC'+str(m2+2))
        if m2>5:
            xlabel('PC1')
        plt.ylim(-3,3)
        plt.xlim(-3,5)
        if m2==1:
            legend(classNames,loc='center left', bbox_to_anchor=(0, 1.3))
            