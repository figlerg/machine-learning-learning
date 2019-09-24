# compute summary statistics for the raw data

from Init import *
# gets variables M,N,X,y, raw_date, df, several others
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
from scipy.linalg import svd
import numpy as np



if __name__ == '__main__':
    cols_x = range(3,11)
    cols_y = range(12,raw_data.shape[1])

    X = raw_data[:,cols_x]
    Y = raw_data[:,cols_y]

    np.corrcoef([x for x in X] + [y for y in Y])
