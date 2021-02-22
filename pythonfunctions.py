import numpy as np
from math import sqrt, exp, log, pi
import scipy.stats as si
from scipy.stats import norm
import pandas as pd
import pandas_datareader as pdr
from pandas_datareader import data as web
from numpy.random import randn
import statsmodels
from statsmodels.tsa.stattools import coint
import quandl
# just set the seed for the random number generator
np.random.seed(107)
import matplotlib.pyplot as plt
#these are mainly finance/trading functions. Change their names accordingly.
# S0= underlying price
# K = strike price
# r= riskless short rate
# T= time to maturity (as percentage of a year)
# sigma = implied volatility
# C0 = initial options call price
# P0= initial options put price

#we define the following from Black-Scholes
def newton_vol_call(S, K, T, C, r, sigma):
    # S: spot price
    # K: strike price
    # T: time to maturity
    # C: Call value
    # r: interest rate
    # sigma: volatility of underlying asset

    d1 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    fx = S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0) - C

    vega = (1 / np.sqrt(2 * np.pi)) * S * np.sqrt(T) * np.exp(-(si.norm.cdf(d1, 0.0, 1.0) ** 2) * 0.5)

    tolerance = 0.000001
    x0 = sigma
    xnew = x0
    xold = x0 - 1

    while abs(xnew - xold) > tolerance:
        xold = xnew
        xnew = (xnew - fx - C) / vega

        return abs(xnew)
#print(newton_vol_call(25, 20, 1, 7, 0.05, 0.25))
#print(dict)
#df1 = pd.DataFrame({a1: np.random.rand(3,1)})
#df2 = pd.DataFrame({a2:np.random.rand(3,1)})
def beta(x,y):
    regr = linear_model.LinearRegression()
    x_constant = pd.concat([x, pd.Series([1] * len(x), index=x.index)], axis=1)
    regr.fit(x_constant, y)
    beta = regr.coef_[0]
    alpha = regr.intercept_
    return beta
def zscore(series):
    return (series-series.mean())/np.std(series)
def reg(x,y):
    regr = linear_model.LinearRegression()
    x_constant = pd.concat([x, pd.Series([1] * len(x), index=x.index)], axis=1)
    regr.fit(x_constant, y)
    beta = regr.coef_[0]
    alpha = regr.intercept_
    spread = y - x * beta - alpha
    return spread

def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.02:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs
def backtestdatasourcetry(): #note requires find_cointegrated_pairs(data)
    from backtester.dataSource.yahoo_data_source import YahooStockDataSource
    from datetime import datetime

    startDate = '2007/12/01'
    endDate = '2017/12/01'
    cachedFolderName = 'yahooData/'  # the folder to which the info is downloaded
    dataSetId = 'testPairsTrading'  # keep this
    instrumentIds = ['SPY', 'AAPL', 'ADBE', 'EBAY', 'MSFT', 'QCOM',
                     'HPQ', 'JNPR', 'AMD', 'IBM']
    ds = YahooStockDataSource(cachedFolderName=cachedFolderName,
                              dataSetId=dataSetId,
                              instrumentIds=instrumentIds,
                              startDateStr=startDate,
                              endDateStr=endDate,
                              event='history')
    data = ds.getBookDataByFeature()['adjClose']

    scores, pvalues, pairs = find_cointegrated_pairs(data)
    import seaborn
    m = [0, 0.2, 0.4, 0.6, 0.8, 1]
    seaborn.heatmap(pvalues, xticklabels=instrumentIds,
                    yticklabels=instrumentIds, cmap='RdYlGn_r'
                    , mask=(pvalues >= 0.98)
                    )
    # plt.show()
    print(pairs)
    # print(data)
    Stock1 = str(input("Enter the first stock in the best pair:"))
    print(Stock1)
    Stock2 = str(input("Enter the second stock in the best pair:"))
    print(Stock2)

    S1 = data[Stock1]
    S2 = data[Stock2]
    score, pvalue, _ = coint(S1, S2)
    print('The pvalue is :', pvalue)
    ratios = S1 / S2
    print('The price ratios are:', ratios)
    # ratios.plot()
    plt.axhline(ratios.mean())
    plt.legend([' Ratio'])
    # plt.show()
def pairtry():
    from backtester.dataSource.yahoo_data_source import YahooStockDataSource
    from datetime import datetime
    startDateStr = '2007/12/01'
    endDateStr = '2017/12/01'
    cachedFolderName = 'yahooData/'
    dataSetId = 'testPairsTrading'
    instrumentIds = ['SPY', 'AAPL', 'ADBE', 'SYMC', 'EBAY', 'MSFT', 'QCOM',
                     'HPQ', 'JNPR', 'AMD', 'IBM']
    ds = YahooStockDataSource(cachedFolderName=cachedFolderName,
                              dataSetId=dataSetId,
                              instrumentIds=instrumentIds,
                              startDateStr=startDateStr,
                              endDateStr=endDateStr,
                              event='history')
    data = ds.getBookDataByFeature()['Adj Close']
    data.head(3)
    scores, pvalues, pairs = find_cointegrated_pairs(data)
    import seaborn
    m = [0, 0.2, 0.4, 0.6, 0.8, 1]
    seaborn.heatmap(pvalues, xticklabels=instrumentIds,
                    yticklabels=instrumentIds, cmap='RdYlGn_r',
                    mask=(pvalues >= 0.98))
    plt.show()
    print(pairs)
def bookpairtry():
    df = web.DataReader(name=['SPY','GOOG','AAPL','MSFT','EBAY']
                        , data_source='yahoo', start='2018-1-1')
    px = df[['Adj Close']]

    px.columns = tickers
    corr = px.corr().unstac().sort_values().drop_duplicates()
    print(corr)
    return px
def checknan():
    check = df.isnull().values.any()
    return check
#print(bookpairtry())
def panelpairtry():
    import yfinance as yf
    yf.pdr_override()
    # for i in range(100):
    #    my_objects.append(MyClass(i))

    # later

    # for obj in my_objects:
    #    print(obj.number)

    np.random.seed(1738)
    startDate = '2019-01-03'
    endDate = '2020-06-26'
    df = pdr.get_data_yahoo("SPY", start="2017-01-01", end="2020-06-30")
    print(df)
    today = date.today()

    # class Tickers(object):
    #    def __init__(self, number):
    #        self.number = number
    # TODO: pxRatios, returns, totalCum returns, beta'ed cumRetSpread, 20day SMA
    # Residual(spread - SMA), zScore, Forward Ret
    # ALSO, v important, find the cointegrated pairs and do all of that ^ on them
    my_tickers = ['GOOG', 'MSFT', 'EBAY']  # ,SPY,AAPL
    # ,'QCOM','HPQ','JNPR','AMD','IBM']

    panels = {i: pdr.data.get_data_yahoo(i, start='2019-01-03'
                                         , end='2020-06-26') for i in my_tickers}
    # for i in my_tickers:
    #    i = pd.DataFrame({i + ' aClose': panels[i]['Adj Close']})

    # print(i)

    px = pd.concat([panels[i]['Adj Close'] for i in my_tickers], axis=1
                   , names=my_tickers, ignore_index=True)
    px.columns = my_tickers

    # pxdata = pd.concat([i for i in (pairs and my_tickers)], axis = 1, ignore_index = True)

    # for i in range(len(my_tickers)):
    #    my_tickers.append(Tickers(i))

    # for symbl in my_tickers:
    #    print(symbl)

    # def tickerdata(ticker):
    #    stock_data = pdr.data.get_data_yahoo(ticker,
    #                                         start = startDate,
    #                                         end = endDate)
    #    print(stock_data)
    #   tickerdata(my_tickers[i])
def quandltry():
    quandl.ApiConfig.api_key = 'eBVoz67PdMxh-o-42uHM'
    # data = quandl.get("AAPL/SPY/EBAY/MSFT")
    tickers = ['SPY', 'MSFT', 'EBAY', 'ADBE',
               'AAPL', 'AMD']

    panels = {i: quandl.get('XNYS/' + i, paginate=True,
                            ticker=i
                            , qopts={'columns': [i, 'date', 'close']},
                            date={'gte': '2015-12-31', 'lte': '2020-07-06'}) for i in tickers}
    print(panels)
    data1 = quandl.get_table('SHARADAR/SEP', paginate=True,
                             ticker=tickers
                             , qopts={'columns': ['ticker', 'date', 'close']},
                             date={'gte': '2015-12-31', 'lte': '2020-07-06'})
    data2 = data1.set_index('date')  # start cleaning by changing index to date
    data = data2.pivot(columns='ticker')  # pivoting the columns so that the tickers are the headers

    print(data.tail())  # checks out
    print(type(data1))  # we good cuz it's a DF
