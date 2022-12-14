import scipy as sp
from data import *

############################
### MAXIMUM SHARPE RATIO ###
############################

def negativeSR(weights, meanReturns, covMatrix, riskFreeRate=0):
    pReturns, pStd = portfolioPerformance(weights, meanReturns, covMatrix)
    return - (pReturns - riskFreeRate) / pStd


def maxSR(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0, 1)):
    """
    Minimize the negative SR, by altering the weights of the portfolio
    """
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    x0 = np.array(numAssets * [1. / numAssets])
    results = sp.optimize.minimize(negativeSR, x0=x0, args=args, method='SLSQP',
                                   bounds=bounds, constraints=constraints)
    return results


##################################
### MINIMUM PORTFOLIO VARIANCE ###
##################################


def portfolioVariance(weights, meanReturns, covMatrix):
    return portfolioPerformance(weights, meanReturns, covMatrix)[1]


def minimizeVariance(meanReturns, covMatrix, constraintSet=(0, 1)):
    """
    Minimize the portfolio variance by altering the weights/allocation of assets in the portfolio
    """
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    x0 = np.array(numAssets * [1. / numAssets])
    result = sp.optimize.minimize(portfolioVariance, x0=x0, args=args, method='SLSQP',
                                  bounds=bounds, constraints=constraints)
    return result


if __name__ == '__main__':
    stockList = ['CBA', 'BHP', 'TLS']
    stocks = [stock + '.AX' for stock in stockList]

    endDate = dt.datetime.now()
    startDate = endDate - dt.timedelta(days=365)

    weights = np.array([0.3, 0.3, 0.4])

    meanReturns, covMatrix = getData(stocks, start=startDate, end=endDate)
    returns, std = portfolioPerformance(weights, meanReturns, covMatrix)

    result = maxSR(meanReturns, covMatrix)
    maxSR, maxWeights = result['fun'], result['x']
    print(-maxSR, maxWeights)

    result = minimizeVariance(meanReturns, covMatrix)
    minVar, minVarWeights = result['fun'], result['x']
    print(minVar, minVarWeights)
