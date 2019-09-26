# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:26:57 2019

@author: louis
"""

from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
from scipy.linalg import svd

import numpy as np
import pandas as pd

filename = 'coil_river_data/analysis.csv'
df = pd.read_csv(filename)
df = df.dropna()

raw_data = df.get_values()

cols = range(3, 11)

X_num = np.asarray(raw_data[:, cols]).astype(np.float64)
X_cat = raw_data[:, range(3)]  # TODO translate this to numbers (Louis!)

X = np.concatenate([X_cat, X_num], 1)
Y = raw_data[:, range(11, 18)]

N, M = X.shape


numerical_normalized = X_num - np.ones((N,1))*X_num.mean(0) * (1/np.std(X_num, 0))
# TODO convert to one out of k (Louis!)
# categorical_oook = ? # "one out of k"

X_normalized = 'not yet implemented'
# X_normalized = np.concatenate([categorical_oook, numerical_normalized], 1)

print('data set up in Init.py')