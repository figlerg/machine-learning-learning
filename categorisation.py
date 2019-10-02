# function to compute classLabels, classNames and y in Init.py

import numpy as np


def classHelper(x, m):
    # just sorts into zeros, <m, >m
    if x == 0:
        return 0
    elif x != 0 and x <= m:
        return 1
    elif x > m:
        return 2


def categorisation(df, target_attribute, mode=0):
    """
    :param df: whole data frame
    :param target_attribute: will categorize this attribute
    :param mode: 0 means we sort into zero values and for the rest: <M, >M (M...median of rest)
    :return: y,classLabels,classNames like in 2_1_1
    """
    column = df[target_attribute]
    not_zero = column[column != 0.0]
    median_rest = not_zero.quantile(0.5)

    # classHelper defined above, sorts into new labels
    y = np.array([classHelper(x, median_rest) for x in column])

    classLabels = [0, 1, 2]
    #classNames = ['0', '<M & != 0', '>M']
    classNames = ['0','0 < '+target_attribute + ' < median(' + target_attribute + ')', target_attribute + ' > median(' + target_attribute + ')']
    return y, classLabels, classNames
