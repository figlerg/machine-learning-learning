# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:37:02 2019

@author: louis
"""

from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
from scipy.linalg import svd

import numpy as np
import pandas as pd

filename = 'coil_river_data/analysis.csv'
df = pd.read_csv(filename)
df=df.dropna()

raw_data = df.get_values()
cols = range(3, 18)

X = np.asarray(raw_data[:, cols]).astype(np.float64)

CORR=np.corrcoef([X[:][i] for i in range(len(X[0]))])
COV = np.cov([X[:][i] for i in range(len(X[0]))])

# print(CORR.shape)
# corr_slice = df2[range(7)][range(8, 14)

print(len(X))

corr_matrix=pd.DataFrame(CORR, columns=df.columns[3:], index=df.columns[3:])
cov_matrix = pd.DataFrame(COV, columns=df.columns[3:], index=df.columns[3:])

mean = X.mean(0)
sd = X.std(axis=0,ddof=1)
labels = list(df.columns)[3:-1]
medians = [np.median(df[name].values) for name in labels] # for num values
# TODO ranges
# OR
stats = df.describe() #select interesting ones

