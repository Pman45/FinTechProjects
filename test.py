import numpy as np
import pandas as pd
import pandas_datareader as pdr
from pandas_datareader import data as web
import statsmodels
from statsmodels.tsa.stattools import coint
import yfinance as yf


# just set the seed for the random number generator

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
            if pvalue < 0.06:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs
# download dataframe
tickers = ['SPY' , 'GOOG', 'AAPL', 'MSFT', 'EBAY']
df = web.DataReader(name=tickers, data_source='yahoo', start='2018-1-1', end='2020-7-6')
data = pd.DataFrame(df[['Adj Close']])
data.columns = tickers
import matplotlib.pyplot as plt

print(data.head())
print(type(data))
scores, pvalues, pairs = find_cointegrated_pairs(data)

print(pairs)

Stock1 = str(input("Enter the first stock in the best pair: "))
print(Stock1)
Stock2= str(input("Enter the second stock in the best pair: "))
print(Stock2)
#d1 = web.DataReader(name=Stock1, data_source='yahoo', start='2018-01-1', end='2020-06-30')
#d2 = web.DataReader(name=Stock2, data_source='yahoo', start='2018-01-1', end='2020-06-30')
S1 = data[[Stock1]]
S2 = data[[Stock2]]
score, pvalue, _ = coint(S1, S2)
print('The pvalue is :' ,pvalue)
ratios = S1 / S2
print('The price ratios are:', ratios)
#ratios.plot()
plt.axhline(ratios.mean())
plt.legend(['Ratio'])
#plt.show()


