import numpy as np
import pandas as pd
import statsmodels
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import scipy.stats as sci
from sklearn import linear_model
import pandas_datareader as pdr
from pandas_datareader import data as web
from pandas.testing import assert_frame_equal
import yfinance as yf
from datetime import datetime
from datetime import date
import seaborn
yf.pdr_override()
np.random.seed(108)
startDate = '2019-01-03'
endDate = '2020-06-26'
my_tickers = ['SPY','MSFT','EBAY','ADBE',
              'AAPL', 'AMD']
              #'HPQ','JNPR','IBM''QCOM']

header_names = ['Date','rOpen', 'rHigh', 'rLow','rClose', 'rVolume'
    , 'flag1', 'flag2', 'aOpen', 'aHigh', 'aLow', 'aClose', 'aVolume']

df1 = pd.read_csv('/Users/piersonmichalak/Desktop/SPY.csv',
                  header=None, skiprows=0, names = header_names)
df1.reset_index(drop=True, inplace=True)
df2 = pd.read_csv('/Users/piersonmichalak/Desktop/MSFT.csv',
                  header=None, skiprows=0, names = header_names)
df2.reset_index(drop=True, inplace=True)
df3 = pd.read_csv('/Users/piersonmichalak/Desktop/EBAY.csv',
                  header=None, skiprows=0, names = header_names)
df3.reset_index(drop=True, inplace=True)
df4 = pd.read_csv('/Users/piersonmichalak/Desktop/ADBE.csv',
                  header=None, skiprows=0, names = header_names)
df4.reset_index(drop=True, inplace=True)
df5 = pd.read_csv('/Users/piersonmichalak/Desktop/AAPL.csv',
                  header=None, skiprows=0, names = header_names)
df5.reset_index(drop=True, inplace=True)
df6 = pd.read_csv('/Users/piersonmichalak/Desktop/AMD.csv',
                  header=None, skiprows=0, names = header_names)
df6.reset_index(drop=True, inplace=True)
list = [df1,df2,df3,df4,df5,df6]
for i in list:
    i.set_index('Date',inplace = True)

SPY = df1[['aClose']][df1.index >= '2017-01-03' ]
MSFT = df2[['aClose']][df2.index >= '2017-01-03' ]
EBAY = df3[['aClose']][df3.index >= '2017-01-03' ]
ADBE = df4[['aClose']][df4.index >= '2017-01-03' ]
AAPL = df5[['aClose']][df5.index >= '2017-01-03' ]
AMD = df6[['aClose']][df6.index >= '2017-01-03' ]


data = pd.concat([SPY,MSFT,EBAY,ADBE,
              AAPL, AMD], axis = 1)
data.columns = my_tickers
print(data.head())
data.corr().unstack().sort_values().drop_duplicates()

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
            if pvalue < 0.1:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs
#NEED PRICE RATIO, MA, SCATTER
scores, pvalues, pairs = find_cointegrated_pairs(data)
m = [0,0.2,0.4,0.6,0.8,1]
seaborn.heatmap(pvalues, xticklabels=my_tickers,
                yticklabels=my_tickers, cmap='RdYlGn_r',
                mask = (pvalues >= 0.98))

#plt.show()
print(data.corr())
#print(data.corr().max())
print(pairs)
Stock1 = input("Enter the first stock in the best pair: ")
print(Stock1)
Stock2 = input("Enter the second stock in the best pair: ")
print(Stock2)


#pairs = pd.concat([Stock1,Stock2],axis = 1)
#for i in list:
 #   for j in my_tickers:
        #if list.index(i) == my_tickers.index(j):
  #         j = i[['aClose']]

#    print(j)
