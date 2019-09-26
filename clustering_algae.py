from Init import df, raw_data, X, X_normalized, Y, N, M, cols

import numpy as np


## normal quantiles!
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
Q25 = df[name].quantile(0.25)
Q50 = df[name].quantile(0.5)
Q75 = df[name].quantile(0.75)

index = list(df.columns).index(name)

attributeNames = np.asarray(df.columns[cols])
classLabels = raw_data[:, index]  # TODO this is weird
classNames = ['0', 'below Q50', 'Q50 and higher']


# classDict = dict(zip(classNames,range(len(classNames))))

# classDict = dict(zip(classNames,range(len(classNames))))
def clas(x, q25, q50, q75):
    res = 0
    if x <= q25:
        return 0
    elif x > q25 and x <= q50:
        return 1
    elif x > q50 and x <= q75:
        return 2
    else:
        return 3

def clas_2(x, q50):
    if x == 0:
        return 0
    elif x != 0 and x <= q50:
        return 1
    elif x > q50:
        return 2


# y = np.array([classDict[cl] for cl in classLabels])

y = np.array([clas_2(x, Q50) for x in classLabels])
