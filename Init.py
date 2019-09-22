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
df=df.dropna()

# raw_data = df.get_values() # apparently this is outdated and gives me a warning (Felix)
raw_data = df.values # does the exact same thing

# columns 3-11 are the CC, therefore our x values
cols = range(3, 11)
X = np.asarray(raw_data[:, cols]).astype(np.float64)

# This is specifically for clustering (look at 1 specific category)
# we sort by category {0:season, 1:river_size, 2:river_speed}
attributeNames = np.asarray(df.columns[cols])
classLabels = raw_data[:,2]
classNames = np.unique(classLabels)

classDict = dict(zip(classNames,range(len(classNames))))

y = np.array([classDict[cl] for cl in classLabels])

N, M = X.shape

C = len(classNames)
