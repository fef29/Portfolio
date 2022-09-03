import scipy as sp
from data import *


def negativeSR(weights, meanReturns, covMatrix, riskFreeRate=0):
    pReturns, pStd = portfolioPerformance(weights, meanReturns, covMatrix)
    return - (pReturns - riskFreeRate) / pStd


def maxSR(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0,1)):
    """
    Minimize the negative SR, by altering the weights of the portfolio
    """
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    x0 = np.array(numAssets * [1./numAssets])
    results = sp.optimize.minimize(negativeSR, x0=x0, args=args, method='SLSQP',
                                   bounds=bounds, constraints=constraints)
    return results


if __name__ == '__main__':
    stockList = ['CBA', 'BHP', 'TLS']
    stocks = [stock + '.AX' for stock in stockList]

    endDate = dt.datetime.now()
    startDate = endDate - dt.timedelta(days=365)

    weights = np.array([0.3, 0.3, 0.4])

    meanReturns, covMatrix = getData(stocks, start=startDate, end=endDate)
    returns, std = portfolioPerformance(weights, meanReturns, covMatrix)

    result = maxSR(meanReturns, covMatrix)
    maxSR, maxWeights = - result['fun'], result['x']
    print(maxSR, maxWeights)