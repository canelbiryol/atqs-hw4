'''
Created on May 9, 2018

@author: Michael
'''
import unittest
from StochasticLQ.SingleStockLQ import SingleStockLQ


class Test_SingleSTockLQ(unittest.TestCase):


    def testSingleStock(self):
        X = SingleStockLQ(0.22, 25000000, 1e-6, 1/252, 7, 0)
        print('The optimal trade is',X.getLiquidationTrade(1000000, 0, 1)[0])
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testSingleStock']
    unittest.main()