#from sklearn import linear_model
import numpy as np 
from scipy.optimize import lsq_linear, curve_fit

def load_data (filename_x='data_x.txt', filename_y='data_y.txt'):
    x = np.loadtxt (filename_x)
    x = np.array (x)

    y = np.loadtxt (filename_y)
    y = np.array (y)

    return x,y 

def model (xdata,c1,c2, c3):
    x = xdata [0]
    y = xdata [1]
    z = xdata [2]
    return ( c1*x + c2*y + c3*z )
def regression (xdata, ydata): 

        #using bayesian ridge regression 
        #reg = linear_model.BayesianRidge ()
        #reg.fit (xdata, ydata)
        #return reg.coef_

        #using multiple regression from scipy 
        #XDATA = (X,Y,Z)
        popt, pcov = curve_fit (model, xdata, ydata )
        popt   = np.array (popt)
        pcov = np.array (pcov)
        print ("Optimal parameters for regression model:\n ", popt)
        print("Covariance: \n", pcov)
        r_squared = np.sqrt ( np.diag (pcov))
        print (r_squared)
        return popt
        #print (xdata)
        #print (ydata)
        #coeff, cost, fun, optim, mask, nit, _ = lsq_linear (xdata, ydata, bounds=(-30,30), verbose=2)
        #return coeff 

x = [ np.linspace (-10,10), np.linspace(-5,5), np.linspace(-2,2) ]
y = np.linspace (-50,51)

def main():
    pcoeff = regression (model, xdata, ydata)
    print (popt)
    print (pcoeff)

    return popt,

