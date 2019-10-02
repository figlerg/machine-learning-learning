import matplotlib.pyplot as plt
from Init import *

from scipy.linalg import svd

U,S,Vh = svd(X_normalized,full_matrices=False)
V=Vh.T

pcs = list(range(10))
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .07
r1 = np.arange(1,11)
r2 = np.arange(1,9)

plt.subplot(2,1,1)
for i in pcs:
    plt.bar(r1-0.245+i*bw, V[0:10,i], width=bw)
    plt.xticks(r1+bw, attributeNames[0:10])
    plt.yticks(np.arange(-0.8, 0.61, 0.2))
    plt.xlabel('Attributes')
    plt.ylabel('Comp. coeff.')
    plt.legend(legendStrs, loc='centre left', bbox_to_anchor=(1,1))
    plt.title('FOIL algae: PCA Component Coefficients')
    

plt.subplot(2,1,2)
for j in pcs:    
    plt.bar(r2-0.245+j*bw, V[10:19,j], width=bw)
    plt.xticks(r2+bw, attributeNames[10:19])
    plt.yticks(np.arange(-0.8, 0.61, 0.2))
    plt.xlabel('Attributes')
    plt.ylabel('Comp. coeff.')
plt.show()
