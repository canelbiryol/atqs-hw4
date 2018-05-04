'''
Created on Apr 28, 2018

@author: canelbiryol
'''
import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import adfuller
from  itertools import combinations

sig_level = '1%' #significance level for the ADF Test
window_size = 40

# Loading data
data_path = './CointData.csv'

def unitRootTest(df):
    for col in df.columns.values:
        result = adfuller(df[col])
        ADF_stat = result[0]
        if ADF_stat < result[4].get(sig_level): #reject the null, this means no unit root I(0) for this significance level, data is stationary
            print('No Unit Root, stationary')
            print(col)    
        else:
            df = df.drop(columns=[col]) # drop the I(1) non-stationary time series
    return df

def calculateGamma(x, y, window_size):
    df_coint = pd.DataFrame()
    df_coint['x'] = x 
    df_coint['y'] = y
    df_coint['xy'] = df_coint['x'] * df_coint['y']
    df_coint['x^2'] = df_coint['x']**2
    df_coint['x_t-1'] = df_coint['x'].shift(1)
    df_coint['y_t-1'] = df_coint['y'].shift(1)
    df_coint['m'] = ((pd.rolling_mean(df_coint['x'], window_size) * pd.rolling_mean(df_coint['y'], window_size)) 
    - (pd.rolling_mean(df_coint['xy'], window_size) )) / (pd.rolling_mean(df_coint['x^2'], window_size) - pd.rolling_mean(df_coint['x'], window_size)**2 )
    df_coint['b'] = pd.rolling_mean(df_coint['y'], window_size) - df_coint['m'] * pd.rolling_mean(df_coint['x'], window_size)
    df_coint['gamma_numerator'] = (df_coint['y'] - (df_coint['m'] * df_coint['x'] + df_coint['b'])) * (df_coint['y_t-1'] - (df_coint['m'] * df_coint['x_t-1'] + df_coint['b']))
    df_coint['gamma_denominator'] = ((df_coint['y_t-1'] - (df_coint['m'] * df_coint['x_t-1'] + df_coint['b'])))**2
    df_coint['gamma'] = (df_coint['gamma_numerator'].rolling(window_size).sum()) / (df_coint['gamma_denominator'].rolling(window_size).sum())
    return df_coint
    
if __name__ == '__main__':
    df = pd.read_csv(data_path, header=None)
    #print(len(df.columns))

    df = unitRootTest(df)
             
    #print(df.head())
    
    pairs = list(combinations(df.columns,2))
    

    for pair in pairs:
        print('{}-{}'.format(pair[0], pair[1]))
        x = df[pair[0]]
        y = df[pair[1]]
        df_coint = calculateGamma(x, y, window_size)
         
        df_coint.to_csv('./results/{}_{}.csv'.format(pair[0], pair[1]))
     
    # print(df_coint)
   
    

    
#     for col in df.columns.values:  #or edit this for a subset of columns first
#         adf_results[col] = tsa.adfuller(df[col])
  
