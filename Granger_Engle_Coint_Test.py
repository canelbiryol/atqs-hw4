'''
Created on Apr 28, 2018

@author: canelbiryol
'''
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as tsa

from numpy.linalg import LinAlgError
from scipy import stats

from statsmodels.regression.linear_model import OLS, yule_walker
from statsmodels.tools.tools import add_constant, Bunch
from statsmodels.tsa.tsatools import lagmat, lagmat2ds, add_trend
from statsmodels.tsa.adfvalues import mackinnonp, mackinnoncrit
from statsmodels.tsa._bds import bds
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tools.sm_exceptions import InterpolationWarning, MissingDataError

SQRTEPS = np.sqrt(np.finfo(np.double).eps)

# Loading data
data_path = './CointData.csv'

def linreg(x, y):
    # y=x*alpha+beta
    window_size = len(x)
    
    x2=np.power(x,2)
    xy=x*y
    window = np.ones(int(window_size))
    a1=np.convolve(xy, window, 'full')*window_size
    a2=np.convolve(x, window, 'full')*np.convolve(y, window, 'full')
    b1=np.convolve(x2, window, 'full')*window_size
    b2=np.power(np.convolve(x, window, 'full'),2)
    alphas=(a1-a2)/(b1-b2)
    betas=(np.convolve(y, window, 'full')-alphas*np.convolve(x, window, 'full'))/float(window_size)
    alphas=alphas[:-1*(window_size-1)] #numpy array of rolled alpha
    betas=betas[:-1*(window_size-1)]   #numpy array of rolled beta
    return alphas

def coint(y0, y1, trend='c', method='aeg', maxlag=None, autolag='aic',
          return_results=None):
    

    trend = trend.lower()
    if trend not in ['c', 'nc', 'ct', 'ctt']:
        raise ValueError("trend option %s not understood" % trend)
    y0 = np.asarray(y0)
    y1 = np.asarray(y1)
    if y1.ndim < 2:
        y1 = y1[:, None]
    nobs, k_vars = y1.shape
    k_vars += 1   # add 1 for y0

    if trend == 'nc':
        xx = y1
    else:
        xx = add_trend(y1, trend=trend, prepend=False)

    res_co = OLS(y0, xx).fit()

    if res_co.rsquared < 1 - 100 * SQRTEPS:
        res_adf = tsa.adfuller(res_co.resid, maxlag=maxlag, autolag=autolag,
                           regression='nc')
    else:
        import warnings
        warnings.warn("y0 and y1 are (almost) perfectly colinear."
                      "Cointegration test is not reliable in this case.")
        # Edge case where series are too similar
        res_adf = (-np.inf,)

    # no constant or trend, see egranger in Stata and MacKinnon
    if trend == 'nc':
        crit = [np.nan] * 3  # 2010 critical values not available
    else:
        crit = mackinnoncrit(N=k_vars, regression=trend, nobs=nobs - 1)
        #  nobs - 1, the -1 is to match egranger in Stata, I don't know why.
        #  TODO: check nobs or df = nobs - k

    pval_asy = mackinnonp(res_adf[0], regression=trend, N=k_vars)
    return res_adf[0], pval_asy, crit

    
if __name__ == '__main__':
    df = pd.read_csv(data_path, header=None)
    #print(df.head())
    print(linreg(df[0], df[1]))
    print(tsa.adfuller(df[0]))
    
#     for col in df.columns.values:  #or edit this for a subset of columns first
#         adf_results[col] = tsa.adfuller(df[col])
  
