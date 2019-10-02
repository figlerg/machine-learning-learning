from matplotlib.pyplot import boxplot, figure, plot, title, xticks, ylabel, show, legend, savefig
from scipy.linalg import svd
from Init import *

import numpy as np
import pandas as pd
from scipy import stats 

#%%
#Correlation matrix

CORR=np.corrcoef([X_num[:][i] for i in range(len(df.columns[3:]))])
corr_matrix=pd.DataFrame(CORR, columns=df.columns[3:], index=df.columns[3:])

#%%
# basic statistics for continuous variables and dummies

X_stats1 = X_num.copy()
attributeNames2 = list(df.columns[cols])

df2=pd.DataFrame(X_stats1, columns=attributeNames2)

describe=df2.describe()
describe=describe.drop(['count','25%','75%'])
describe=describe.rename(index={'50%':'Median'})
describe=describe.round(1)

X_stats2 = np.c_[autumn, spring, summer, winter, small_river, medium_river, large_river, low_flow, medium_flow, high_flow]
attributeNames3 = ['autumn', 'spring', 'summer', 'winter', 'small_river', 'medium_river','large_river', 'low_flow', 'medium_flow', 'high_flow']
df3=pd.DataFrame(X_stats2, columns=attributeNames3)
test=df3.apply(pd.value_counts)

index=test.columns
zeros=test.values[0]
ones=test.values[1]
df4 = pd.DataFrame({'0': zeros,'1': ones}, index=index)
ax = df4.plot.bar(rot=0,figsize=(13,3))
show()
        
#%%
#Boxplots

figure()
boxplot(numerical_normalized)
xticks(range(1,9),list(df.columns[cols]))
ylabel('normalized occurence [-]')
title('Boxplot')
show()