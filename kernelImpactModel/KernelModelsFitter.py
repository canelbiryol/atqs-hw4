import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class KernelModelFitter(object):
    """
    As in homework 2, reads the stats and then perform the regression on them.
    """

    def __init__(self, repo):

        self._path = repo
        
        # Read stats
        self._ADV = pd.read_csv(self._path + 'ADVal.csv', index_col=0)
        self._Imbalance = pd.read_csv(self._path + 'Imbalance.csv', index_col=0)
        self._vol = pd.read_csv(self._path + 'vol.csv', index_col=0)
        self._VWAPuntil330 = pd.read_csv(self._path + 'VWAP330.csv', index_col=0)
        self._VWAPuntil400 = pd.read_csv(self._path + 'VWAP400.csv', index_col=0)
        self._LoopBack = pd.read_csv(self._path + 'PriceLoopBack.csv', index_col=0)
        self._FirstPrice = pd.read_csv(self._path + 'FirstPrice.csv', index_col=0)
        self._LastPrice = pd.read_csv(self._path + 'LastPrice.csv', index_col=0)
        self._STDLoopBack = pd.read_csv(self._path + 'StdLoopBack.csv', index_col=0)

        # Sort and assign labels
        self._labels = np.array(list(set.intersection(*list(map(lambda x: set(x.columns), [self._ADV, self._Imbalance, self._vol, self._VWAPuntil330, self._VWAPuntil400, self._LoopBack,self._FirstPrice, self._LastPrice, self._STDLoopBack])))))
        self._dates = np.array(self._ADV.index)
        self._ADV, self._Imbalance, self._vol, self._VWAPuntil330, self._VWAPuntil400, self._LoopBack, \
        self._FirstPrice, self._LastPrice, self._STDLoopBack = map(lambda x: x[self._labels].sort_index(ascending=True), [self._ADV, self._Imbalance, self._vol, self._VWAPuntil330, self._VWAPuntil400, self._LoopBack, self._FirstPrice, self._LastPrice, self._STDLoopBack])

        # Market impacts
        self._PermanentImpact = (self._LastPrice - self._FirstPrice) / self._FirstPrice
        self._PermanentImpact330 = (self._VWAPuntil330 - self._FirstPrice) / self._FirstPrice
        self._TemporaryImpact = self._PermanentImpact330 - self._PermanentImpact / 2

    def getTemporaryImpact(self):
        return self._TemporaryImpact

if __name__ == '__main__':
    
    # Initialize a kernel model fitter
    statsPath = '/media/louis/DATA/documents/cours/NYU/SPRING_18/ATQS/atqs-hw4/stats/'
    resPath = statsPath + 'res/'
    kernelFitter = KernelModelFitter(statsPath)
    
    # Read data, get temporary impact. Keep only common tickers
    stockPrices = pd.read_csv(statsPath+'MidpriceFor10min.csv', index_col=[0])
    temporaryImpact = kernelFitter.getTemporaryImpact()
    stockTickers = np.array(list(set.intersection(set(stockPrices.columns), set(temporaryImpact.columns))))
    stockPrices, temporaryImpact = stockPrices[stockTickers].values, temporaryImpact[stockTickers].values
    numberValues = stockPrices.shape[1]
    
    # 2 models to fit
    Gt_exp = lambda t, rho: np.exp(-rho * t)
    Gt_power = lambda t, gamma: t ** (-gamma)
    _t = np.array([i for i in range(1, 43)])

    # Parameters fitted, as well as their variances
    GammasPower, RhosExponential, Gammas_STD, RhosExponential_STD = np.empty((64,1)).flatten(), np.empty((64,1)).flatten(), np.empty((64,1)).flatten(), np.empty((64,1)).flatten()

    # Use all 64 observations. Since we use 10 min midprices, each day is splitted in 39.
    for i in range(64):
        
        # Temporary impact
        h = temporaryImpact[i, :]

        # Day t, last price before 3:30
        priceCurDay330 = 39*i+ 34
        # Day t, first price after 3:30
        priceCurDay330400 = 39 * i + 35
        # Day t+1, price at 4:30
        priceNextDay430 = 39*i + 2*39 - 1

        # Compute "empirical" gamma
        gammaToFit = np.zeros((numberValues, len(_t)))
        for j in range(numberValues):
            gammaToFit[j, :] = (stockPrices[priceCurDay330400:priceNextDay430, j] - stockPrices[priceCurDay330, j]) / stockPrices[priceCurDay330, j] / h[j]
        
        # Remove NaN
        gammaToFit = np.nanmedian(gammaToFit, axis=0)
        boolNaN = np.isnan(gammaToFit)
        gammaToFit, Tindex = gammaToFit[~boolNaN], _t[~boolNaN]
        
        # Prepare the input for curve_fit (normalize)
        gammaToFit = gammaToFit - min(gammaToFit)
        gammaToFit = gammaToFit/ max(gammaToFit)

        # gammaToFit against exponential and power laws
        residualsPower = curve_fit(Gt_power, Tindex, gammaToFit, bounds=[0, 1])
        residualsExponential = curve_fit(Gt_exp, Tindex, gammaToFit, bounds=[0, np.inf])
        
        # Store  the values and the residuals
        GammasPower[i] = residualsPower[0][0]
        RhosExponential[i] = residualsExponential[0][0]
        Gammas_STD[i] = np.sqrt(residualsPower[1])[0, 0]
        RhosExponential_STD[i] = np.sqrt(residualsExponential[1])[0, 0]

    print("Daily values of gamma:\n", GammasPower)
    print("\nDaily values of rhos:\n", RhosExponential)

    # Rho analysis
    plt.scatter(np.arange(len(RhosExponential)), RhosExponential, label=r'$\rho$')
    plt.legend()
    plt.title(r'Change of rho accross days')
    plt.savefig(resPath + 'rho_change.png')
    
    plt.clf()

    plt.hist(np.array(RhosExponential))
    plt.title(r'Distribution of rho')
    plt.savefig(resPath + 'rho_dist.png')
    
    plt.clf()

    # Gamma analysis
    plt.scatter(np.arange(len(GammasPower)), GammasPower, label=r'$\gamma$')
    plt.legend()
    plt.title(r'Change of gamma accross days')
    plt.savefig(resPath + 'gamma_change.png')

    plt.clf()

    plt.hist(np.array(GammasPower))
    plt.title(r'Distribution of gamma')
    plt.savefig(resPath + 'gamma_dist.png')
