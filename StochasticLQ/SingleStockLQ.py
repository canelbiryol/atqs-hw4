'''
Created on May 8, 2018

@author: Michael
'''
import numpy as np

class SingleStockLQ(object):
    '''
    classdocs
    '''


    def __init__(self, vol, ADV, riskAversion, totalTime, n_periods, tradingcosts):
        self.vol = vol
        self.ADV = ADV
        self.riskAversion = riskAversion
        self.T = totalTime
        self.n_periods = n_periods
        self.Lambda = tradingcosts
        self.Pi = 0.314
        self.H = 0.142
        
        self.Sigma = self.vol*self.T/self.n_periods
        self.R = np.array([[-0.5*self.riskAversion*self.Sigma, -0.5], [-0.5, 0]])
        self.Q = -0.5*self.Lambda
        self.S = np.array([[(self.Pi + self.H)/2],[0]])
        self.A = np.array([[1, 0], [0, 0]])
        self.B = np.array([[1],[self.H]])
        
        self.Kt = [None]*n_periods
        self.Kt[n_periods-1] = self.R 
        for p in range(n_periods-1):
            t = n_periods-2-p
            W = self.S+np.matmul(np.transpose(self.A),np.matmul(self.Kt[t+1],self.B))
            Z = np.transpose(self.S)+np.matmul(np.transpose(self.B),np.matmul(self.Kt[t+1],self.A))
            X = np.matmul(W,Z)/(self.Q+np.matmul(np.transpose(self.B),np.matmul(self.Kt[t+1],self.B)))
            self.Kt[t] = self.R+np.matmul(np.transpose(self.A),np.matmul(self.Kt[t+1],self.A))-X
        

    def getLiquidationTrade(self, xt, ht, t):
        X = np.transpose(self.S)+np.matmul(np.transpose(self.B),np.matmul(self.Kt[t],self.A))
        Y = -X/(self.Q+np.matmul(np.transpose(self.B),np.matmul(self.Kt[t],self.B)))
        return np.dot(Y,np.array([xt, ht]))
        