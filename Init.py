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

raw_data = df.get_values() 

cols = range(3, 11) 

X = np.asarray(raw_data[:, cols]).astype(np.float64)

attributeNames = np.asarray(df.columns[cols])
classLabels = raw_data[:,0] 
classNames = np.unique(classLabels)

classDict = dict(zip(classNames,range(len(classNames))))

y = np.array([classDict[cl] for cl in classLabels])

N, M = X.shape

C = len(classNames)
