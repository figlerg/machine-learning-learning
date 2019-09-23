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

attributeNames = ['season', 'river_size', 'flow_vel','CC1','CC2','CC3','CC4','CC5','CC6','CC7','CC8','AG1','AG2','AG3','AG4','AG5','AG6','AG7']

df = pd.read_csv((filename),names=attributeNames)
df=df.dropna()

raw_data = df.get_values() 

cols = range(3, 11) 

X = np.asarray(raw_data[:, cols]).astype(np.float64)

attributeNames = np.asarray(df.columns[cols])
classLabels = raw_data[:,0] 
classNames = np.unique(classLabels)

classDict = dict(zip(classNames,range(len(classNames))))

y = np.array([classDict[cl] for cl in classLabels])
#y = np.zeros(len(X))
y#[df.AG6>np.median(df.AG6)] = 1

N, M = X.shape

C = len(classNames)
