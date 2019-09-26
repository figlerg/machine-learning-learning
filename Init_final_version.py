# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:26:57 2019

@author: louis
"""

from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
from scipy.linalg import svd

import numpy as np
import pandas as pd
import categoric2numeric

filename = 'coil_river_data/analysis.csv'
df = pd.read_csv(filename)
df = df.dropna()

raw_data = df.get_values()

cols = range(3, 11)

X_num = np.asarray(raw_data[:, cols]).astype(np.float64)

Y = raw_data[:, range(11, 18)]


numerical_normalized = np.array(X_num - np.ones((167,1))*X_num.mean(0) * (1/np.std(X_num, 0)))

season=categoric2numeric.categoric2numeric([raw_data[i][0] for i in range(len(raw_data))])
autumn=np.array([x[0] for x in season[0]])
spring=np.array([x[1] for x in season[0]])
summer=np.array([x[2] for x in season[0]])
winter=np.array([x[3] for x in season[0]])

river_size=categoric2numeric.categoric2numeric([raw_data[i][1] for i in range(len(raw_data))])
large_river=np.array([x[0] for x in river_size[0]])
medium_river=np.array([x[1] for x in river_size[0]])
small_river=np.array([x[2] for x in river_size[0]])

flow_vel=categoric2numeric.categoric2numeric([raw_data[i][2] for i in range(len(raw_data))])
high_flow=np.array([x[0] for x in flow_vel[0]])
low_flow=np.array([x[1] for x in flow_vel[0]])
medium_flow=np.array([x[2] for x in flow_vel[0]])

X_normalized = np.c_[autumn, spring, summer, winter, large_river, medium_river, small_river, high_flow, low_flow, medium_flow, numerical_normalized]

N, M = X_normalized.shape

attributeNames = ['autumn', 'spring', 'summer', 'winter', 'large_river', 'medium_river', 'small_river', 'high_flow', 'low_flow', 'medium_flow'] + list(df.columns[cols])

#Todo : classLabels, classNames and y : FELIX

print('data set up in Init.py')