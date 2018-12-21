import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from pprint import pprint


def correlation(x,y):
    """
    takes x and y 1-D arrays
    returns value for off-diagonal correlation coefficient
    """
    return np.corrcoef(x, y)[0,1]

def analyze_corrs(data):
    full_X = data.data
    y = data.target
    corrs = [correlation(cols, y) for cols in full_X.T]
    corrs_names = [(f, c) for f,c in zip(data.feature_names, corrs)]
    pprint(corrs_names)
    return True


def diagnose_residuals(x,y, model):
    """ 
    takes x, y and fitted linear regression model
    plots x and residuals 
    plots histograms of residuals
    prints R2 score
    return True
    """
    pass


def main():
    """
    main script
    """
    data = load_boston()
    full_X = data.data
    y = data.target
    print('shape de full_X : ', full_X.shape)
    print('shape de y : ', y.shape)
    print('\n \n') #double new line
    
    descr = False #disable DESCR print, put True to enable 
    if descr : print(data.DESCR)

    analyze_corrs(data)

if __name__ == '__main__':
    main()
