# exercise 2.2.4

# (requires data structures from ex. 2.2.1)
import matplotlib.pyplot as plt
from Init import *

from scipy.linalg import svd

U,S,Vh = svd(X_normalized,full_matrices=False)
V=Vh.T

pcs = list(range(10))
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
for i in pcs:
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('FOIL algae: PCA Component Coefficients')
# plt.savefig('influence_plot.jpg')
plt.show()

## TODO clean this copy paste stuff out, probably unnecessary:
# Inspecting the plot, we see that the 2nd principal component has large
# (in magnitude) coefficients for attributes A, E and H. We can confirm
# this by looking at it's numerical values directly, too:
# print('PC2:')
# print(V[:,1].T)

# How does this translate to the actual data and its projections?
# Looking at the data for water:

# Projection of water class onto the 2nd principal component.
# all_water_data = Y[y==4,:]

# print('First water observation')
# print(all_water_data[0,:])
#
# # Based on the coefficients and the attribute values for the observation
# # displayed, would you expect the projection onto PC2 to be positive or
# # negative - why? Consider *both* the magnitude and sign of *both* the
# # coefficient and the attribute!
#
# # You can determine the projection by (remove comments):
# print('...and its projection onto PC2')
# print(all_water_data[0,:]@V[:,1])
# # Try to explain why?