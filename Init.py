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

X_num = np.asarray(raw_data[:, cols]).astype(np.float64)
X_cat = raw_data[:,range(3)] # TODO translate this to numbers/ one out of k dummies

X = np.concatenate([X_cat, X_num], 1)


## quantiles!
# name = 'AF1'
# Q25=df[name].quantile(0.25)
# Q50=df[name].quantile(0.5)
# Q75=df[name].quantile(0.75)
#
# attributeNames = np.asarray(df.columns[cols])
# classLabels = raw_data[:,13]
# classNames = ['<Q25','Q25< and <Q50','Q50< and <Q75','>Q75']

## other categories
name = 'AF3'
Q25=df[name].quantile(0.25)
Q50=df[name].quantile(0.5)
Q75=df[name].quantile(0.75)

index = list(df.columns).index(name)

attributeNames = np.asarray(df.columns[cols])
classLabels = raw_data[:,index] # TODO this is weird
classNames = ['0','below Q50','Q50 and higher']

#classDict = dict(zip(classNames,range(len(classNames))))
def clas_2(x,q50):
    if x== 0:
        return 0
    elif x!=0 and x<=q50:
        return 1
    elif x>q50:
        return 2



#classDict = dict(zip(classNames,range(len(classNames))))
def clas(x,q25,q50,q75):
    res=0
    if x<= q25:
        return 0
    elif x>q25 and x<=q50:
        return 1
    elif x>q50 and x<=q75:
        return 2
    else:
        return 3
    
#y = np.array([classDict[cl] for cl in classLabels])

y = np.array([clas_2(x,Q50) for x in classLabels])

N, M = X.shape

C = len(classNames)

print('winter',df[df.season=='winter'].season.count())
print('spring',df[df.season=='spring'].season.count())
print('summer',df[df.season=='summer'].season.count())
print('autumn',df[df.season=='autumn'].season.count())
print('river_size')
print('small_',df[df.river_size=='small_'].river_size.count())
print('medium',df[df.river_size=='medium'].river_size.count())
print('large_',df[df.river_size=='large_'].river_size.count())
print('flow_vel')
print('low___',df[df.flow_vel=='low___'].flow_vel.count())
print('medium',df[df.flow_vel=='medium'].flow_vel.count())
print('high__',df[df.flow_vel=='high__'].flow_vel.count())

