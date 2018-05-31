import numpy as np
from scipy import stats

r"""
Axes are defined for arrays with more than one dimension. 
A 2-dimensional array has two corresponding axes: 
the first running vertically downwards across rows (axis 0), 
and the second running horizontally across columns (axis 1).
"""

def isEmpty(data):
    if (len(data) == 0): raise ValueError(" Empty Array")
    #elif(data == None): raise ValueError(" Empty Array")

def check(kernel):
    switcher = {
        'std':True,
        'ratio':True,
        'mean':True,
        'zscore':True
    }
    return switcher.get(kernel,False)

"""
Analyzes if the combination of shape and axis given is correct
"""
def checkaxis(shape,axis):
    value = True
    rowcount, colcount = shape
    if (axis == 0 and rowcount <=1
            or axis == 1 and colcount <=1):
        print ("Error: cannot make operations on scalars")
        value = False
    return value

def transform(data, axis, retransform = None):
    newdata = data.copy()
    if (axis is None):
        if (retransform is not None):
            newdata = np.reshape(newdata,retransform)
        else:
            rowcount,colcount = data.shape
            newdata = np.reshape(data,rowcount*colcount).T
    elif(axis == 1):
        newdata = newdata.T
    return newdata

def apply_kernel(kernel,column, axis):
    STANDARD_DEV = 'std'
    RATIO = 'ratio'
    MEAN = 'mean'
    ZSCORE = 'zscore'
    
    min = column.min()
    max = column.max()
    mean = column.mean()
    
    if (kernel == STANDARD_DEV):
        stdDev = abs(mean-min) + abs(max-mean)
        newcol = (column - mean) / stdDev
    elif (kernel == RATIO):
        newcol = column/max
    elif (kernel == MEAN):      #"""Mean is still under development"""
        denom = max - min
        newcol = (column - mean)/denom
    elif(kernel == ZSCORE):
        newcol = stats.zscore(column,axis)
    return newcol
        
"""
Normalize data given a base kernel. Shape should be a 2-D array.
Parameters:
    data: array_like
        input array. 
    axis: int or None, optional
        The axis to operate on. Default axis is 0.
    kernel: string or None, optional
        The formula that is going to be use to normalize.
        Supported kernels: standard deviation, and ratio
"""
def normalize(data, axis = 0, kernel = "std"):
    isEmpty(data)
    if (not check(kernel)): return
    data = np.mat(data)
    if (not checkaxis(data.shape,axis)): return
    #transform the matrix given the axis to operate
    newdata = transform(data, axis)
    rowcount,colcount = newdata.shape
    newmatrix = []
    for i in range(colcount):
        colpos = newdata[:,i]  
        newcol = apply_kernel(kernel,colpos, axis)
        newmatrix.append(newcol)
    newmatrix = np.concatenate(newmatrix, axis = 1)
    return transform(newmatrix,axis,data.shape)

"""
Returns a new numpy array with rows, that lacked attributes, eliminated.
The purpose is to get rid of inconsistent data. 
Parameters:
    data: array_like
        input array. 
    return_counts: bool,optional
        If True, also return the number of rows eliminated from data.
    axis: int or None, optional
        The axis to operate on.
"""
def clean_misval(data, return_counts = False, axis =None ):
    if (not check(data)):
        return