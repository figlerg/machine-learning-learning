from Init import *
#from david_main import *
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend, xlim, ylim
from scipy.linalg import svd

#Y  = X_normalized - np.ones((N,1))*X_normalized.mean(0)
#Y1 = Y*(1/np.std(Y,0))
Y1 = X_normalized


U,S,Vh = svd(Y1,full_matrices=False)

V = Vh.T    

Z = Y1 @ V

i = 2
j = 3

colors =['r','y','b']

f = figure()
title('river data: PCA')

for c in range(C):
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5, color=colors[c])
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))


show()

