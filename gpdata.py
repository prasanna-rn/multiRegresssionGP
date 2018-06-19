# should contain global x and y variables, 
# y_pred function to return by x element, 
# method to load data 

import numpy as np 

def getData(filename):
    inputarray = np.loadtxt (filename)
    #print (inputarray)
    x = np.array (inputarray[:,0])
    y = np.array (inputarray[:,1])
    print (x)
    print (y)
    return x,y 
x,y = getData ('data.txt')
def y_obs (xValue):
    location = np.where (x == xValue)
    return y[location]

def y_obs_index (xIndex):
    return y [xIndex]

def generateTestData ():
    values  = range (1, 5000)
    for value in values: 
        results.append (values / 3.1415 ) 