import numpy as np

from backtester.dataSource.yahoo_data_source import YahooStockDataSource
from datetime import datetime

startDateStr = '2006/12/01'
endDateStr = '2020/07/06'
cachedFolderName = 'yahooData/'  # the folder to which the info is downloaded
dataSetId = 'testPairsTrading'  # keep this
instrumentIds = ['SPY', 'AAPL', 'ADBE', 'EBAY', 'MSFT', 'QCOM',
                     'HPQ', 'JNPR', 'AMD', 'IBM']
ds = YahooStockDataSource(cachedFolderName=cachedFolderName,
                            dataSetId=dataSetId,
                            instrumentIds=instrumentIds,
                            startDateStr=startDateStr,
                            endDateStr=endDateStr,
                            event='history')
data = ds.getBookDataByFeature()['Adj Close']
import pandas as pd
import statsmodels
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import scipy.stats as sci
from sklearn import linear_model
#import pandas_datareader as pdr <- these two cause issues for some reason
#from pandas_datareader import data as web

#from datetime import datetime
#from datetime import date


#have plan to request input tickers and then produce an array with all of those tickers
#and make it so that they can stop inputting tickers
#god cointegration function
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
            if pvalue < 0.0709:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs
                              event='history')
data = ds.getBookDataByFeature()['adjClose']

scores, pvalues, pairs = find_cointegrated_pairs(data)
def heatplot():
    import seaborn
    m = [0, 0.2, 0.4, 0.6, 0.8, 1]
    seaborn.heatmap(pvalues, xticklabels=instrumentIds,
                    yticklabels=instrumentIds, cmap='RdYlGn_r'
                    , mask=(pvalues >= 0.98)
                    )
    plt.show()
#heatplot()
print(pairs)
print(data.tail())
Stock1 = str(input("Enter the first stock in the best pair:"))
print(Stock1)
Stock2 = str(input("Enter the second stock in the best pair:"))
print(Stock2)

S1 = data[Stock1]
px1 = pd.DataFrame(S1, index = data.index)
print(type(px1))
S2 = data[Stock2]
px2 = pd.DataFrame(S2, index = data.index)
pxdata = pd.concat([px1,px2], axis = 1)
pxdata.columns = [Stock1,Stock2]
score, pvalue, _ = coint(S1, S2)
print('The pvalue is :', pvalue)
ratios = S1 / S2
#calculate zscore
def zscore(series):
    return (series-series.mean())/np.std(series)
#print(zscore(ratios))
#print('The price ratios are:', ratios)
pxratios = pd.DataFrame(ratios, index = data.index)
pxratios.columns = ['Ratio']

def reg(x,y):
    regr = linear_model.LinearRegression()
    x_constant = pd.concat([x, pd.Series([1] * len(x), index=x.index)], axis=1)
    regr.fit(x_constant, y)
    beta = regr.coef_[0]
    alpha = regr.intercept_
    spread = y - x * beta - alpha
    return spread
def beta(x,y):
    regr = linear_model.LinearRegression()
    x_constant = pd.concat([x, pd.Series([1] * len(x), index=x.index)], axis=1)
    regr.fit(x_constant, y)
    beta = regr.coef_[0]
    alpha = regr.intercept_
    return beta
lp = np.log(pxdata)
x = lp[Stock1]
y = lp[Stock2]
spread = reg(x,y)
beta = beta(x,y)
#gives  adjreturns
def adjretdata():
    adjret = np.zeros((3419,2), np.float64)
    adjret[0][0] = (px1.iat[0,0])/(px1.iat[0,0])
    adjret[0][1] = (px1.iat[0,0])/(px1.iat[0,0])
    for i in range(3418):
        adjret[i][0] = (px1.iat[i,0] / px1.iat[i - 1,0] - 1) * 100  # return1
        adjret[i][1] = (px2.iat[i,0] / px2.iat[i - 1,0] - 1) * 100  # return2
    return adjret
np.random.seed(123)
adjret = adjretdata()
Rets = pd.DataFrame({'AdjRet1':adjret[:,0], 'AdjRet2':adjret[:, 1]},index = data.index)
Zscore = pd.DataFrame(zscore(ratios),index = data.index)
Zscore.columns = ['Ratio Zscore']
CumRet = Rets.cumsum()
CumRet.columns = ['CumRet1','CumRet2']
CumRet.index = data.index
#caluclate beta cum ret spread
def betaspreaddata():
    betaspread = np.zeros((3419,1), np.float64)
    betaspread[0][0]= CumRet.iat[0,0] - beta*CumRet.iat[0,1]
    for j in range(3418):
        betaspread[j][0] = CumRet.iat[j,0] - beta*CumRet.iat[j,1]
    return betaspread #calculate beta cum ret spread
betaspread = betaspreaddata()
BetaSpread = pd.DataFrame({'Beta cum ret spread':betaspread[:,0]},index = data.index)
SMA = BetaSpread.rolling(20).mean()
SMA.columns = ['Spread MA']
SMA.index=data.index
#calculate spread-MA
def differencedata():
    diff = np.zeros((3419,1), np.float64)
    diff[0][0] = BetaSpread.iat[0,0] - SMA.iat[0,0]
    for i in range(3418):
        diff[i][0] = BetaSpread.iat[i,0] - SMA.iat[i,0]

    return diff
diff = differencedata()
Spread_minus_MA = pd.DataFrame({'Spread - MA':diff[:,0]}, index = data.index)
DiffStd = Spread_minus_MA.rolling(30).std()
DiffStd.columns = ['Rolling Std(Spread-MA)']
DiffStd.index = data.index
ratios.plot()
plt.axhline(ratios.mean())
plt.legend(['Ratio'])
output = pd.concat([pxdata,Rets,CumRet,Zscore,BetaSpread,SMA,Spread_minus_MA,DiffStd], axis = 1)
#output.to_excel(r'/Users/piersonmichalak/Desktop/pairtest2.xlsx',sheet_name ='sheet1',index=True)

#def rollingdata():
#    rolling = np.zeros((3419,1), np.float64)
#    for i in range(21):
#        rolling[i][0]=np.nan
#    for j in range(3419-21):
#        rolling[j][0] = BetaSpread.iat[j,0].rolling(20).mean()
#
#    return rolling

#plt.show()

#TODO: total corr, total beta = ret hedge, share hedge, returns, total cum rets
#total beta'ed cumretspread, 20day SMA spread, residual (= spread - SMA)
#z-score, forward returns, the ratio stuff and the graphs he has