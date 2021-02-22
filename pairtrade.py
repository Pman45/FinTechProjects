import numpy as np
#import clr
#from clr import AddReference
#AddReference("System")
#AddReference("QuantConnect.Common")
#AddReference("QuantConnect.Jupyter")
#AddReference("QuantConnect.Indicators")
import pandas as pd
import scipy.stats as sci
import sys
import scipy
from scipy import stats
import statsmodels
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
#from linearmodels.iv import IV2SLS
import seaborn
import matplotlib.pyplot as plt
from sklearn import linear_model
from openpyxl import load_workbook

def cov(a, b):

    if len(a) != len(b):
        return

    a_mean = np.mean(a)
    b_mean = np.mean(b)

    sum = 0

    for i in range(0, len(a)):
        sum += ((a[i] - a_mean) * (b[i] - b_mean))

    return sum/(len(a)-1)

np.random.seed(107)
header_names1 = ['Date','rOpen', 'rHigh', 'rLow','rClose', 'rVolume'
    , 'flag1', 'flag2', 'aOpen', 'aHigh', 'aLow', 'aClose1', 'aVolume']
header_names2 = ['Date','rOpen', 'rHigh', 'rLow','rClose', 'rVolume'
    , 'flag1', 'flag2', 'aOpen', 'aHigh', 'aLow', 'aClose2', 'aVolume']

df1 = pd.read_csv('/Users/piersonmichalak/Desktop/AAPL.csv',header=None, skiprows=0, names = header_names1)
#(1 to 9962), from 1980-12-12 to 2020-06-17
# so updated from 1325 to 9962 should be good
df2 = pd.read_csv('/Users/piersonmichalak/Desktop/MSFT.csv',header=None, skiprows=0, names = header_names2)
#(1 to 8637), from 1986-03-13 to 2020-06-17
df1.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)
#print(df1.iat[1325,11], df2.iat[0,11], np.log(df1.iat[1325,11])/np.log(df2.iat[0,11]))

S1 = 'aClose1'.join('AAPL')
S2 = 'aClose2'.join('MSFT')
datetime = df2[['Date']]
stock1 = df1[['aClose1']][df1.index >= 1325 ]
stock1.reset_index(drop=True, inplace=True)
stock2 = df2[['aClose2']]
stock2.reset_index(drop=True, inplace=True)
stockdata = pd.concat([datetime,stock1,stock2], axis = 1)

symbols = ['AAPL','MSFT']
prices = pd.concat([stock1,stock2],axis = 1)
prices.columns = symbols
lp = np.log(prices)
def reg(x,y):
    regr = linear_model.LinearRegression()
    x_constant = pd.concat([x, pd.Series([1] * len(x), index=x.index)], axis=1)
    regr.fit(x_constant, y)
    beta = regr.coef_[0]
    alpha = regr.intercept_
    spread = y - x * beta - alpha
    return spread

x = lp['AAPL']
y = lp['MSFT']
spread = reg(x,y)
# plot the spread series
spread.plot(figsize =(15,10))
plt.ylabel('spread')
#plt.show()

stockdata.rename(columns={'Date':'Date','aClose1': 'AAPL', 'aClose2':'MSFT'}, inplace=True)
#stock_name_1 = 'AAPL'
#stock_name_2 = 'MSFT'

#print(stockdata)
def beta(x,y):
    regr = linear_model.LinearRegression()
    x_constant = pd.concat([x, pd.Series([1] * len(x), index=x.index)], axis=1)
    regr.fit(x_constant, y)
    beta = regr.coef_[0]
    alpha = regr.intercept_
    return beta

def spreaddata():
    spreadt = np.zeros((8637,5), np.float64)

    spreadt[0][0] = stock1.iat[0,0]#Stock 1 price
    spreadt[0][1] = stock2.iat[0,0] #Stock 2 price
    zerop = prices[['AAPL','MSFT']][prices.index <= 0]

    spreadt[0][2] =  beta(zerop['AAPL'],zerop['MSFT'])#(np.cov(spreadt[0][0].T,spreadt[0][1].T)[0][1])#/(np.var(spread[0][1]))
    print('Initial beta supposedly is', spreadt[0][2])
    n = np.log(spreadt[0][0])/np.log(spreadt[0][1]) #beta = hedge ratio
    spreadt[0][3] = np.log(spreadt[0][0]) - spreadt[0][2]*np.log(spreadt[0][1]) #residual spread
    spreadt[0][4] = (df1.iat[1325,11]/df2.iat[0,11])*spreadt[0][3] #share hedge ratio
    for i in range(1,8637):
        n = np.log(spreadt[0][0]) / np.log(spreadt[0][1])
        spreadt[i][0] = stock1.iat[i,0]# Stock 1 price
        spreadt[i][1] = stock2.iat[i, 0]# Stock 2 price
        #w = spreadt[0:i][0]
        #print(w)
        #z = spreadt[0:i][1]
        #sci.stats.linregress(w, z)
        #slope, intercept, r_value, p_value, std_err = stats.linregress(w, z)
        nthprice = prices[['AAPL','MSFT']][prices.index<= i]
        spreadt[i][2] = beta(nthprice['AAPL'],nthprice['MSFT'])

        #(np.cov(spreadt[i][0].T, spreadt[i][1].T)[1][1]) #/(np.var(spread[i][1]))
        # beta = return hedge ratio
        spreadt[i][3] = np.log(spreadt[i][0]) - spreadt[i][2] * np.log(spreadt[i][1]) #actual spread
        spreadt[i][4] = (spreadt[i][0])/(spreadt[i][1])*(spreadt[i][3])  #share hedge ratio
    return spreadt

def adjretdata():
    adjret = np.zeros((8637,2), np.float64)
    adjret[0][0] = (df1.iat[1325,11])/(df1.iat[1324,11])
    adjret[0][1] = 0
    for i in range(8636):
        adjret[i][0] = (df1.iat[i+1325,11] / df1.iat[i - 1 + 1325,11] - 1) * 100  # return1
        adjret[i][1] = (df2.iat[i,11] / df2.iat[i - 1,11] - 1) * 100  # return2
    return adjret

np.random.seed(123)
spreadt = spreaddata()
spreadt[:, 0].round(10)
adjret = adjretdata()


symbols = ['AAPL','MSFT']
Spr = pd.DataFrame({'Spread':spreadt[:, 3], 'RetHedge':spreadt[:, 2], 'ShareHedge':spreadt[:, 4]})
Ret = pd.DataFrame({'AdjRet1':adjret[:,0], 'AdjRet2':adjret[:, 1]})

def corr():
    mean1 = stock1.iloc[:, 0].mean()
    mean2 = stock2.iloc[:, 0].mean()
    std1 = stock1.iloc[:, 0].std()
    std2 = stock2.iloc[:, 0].std()
    corr = ((stock1.iloc[:, 0]*stock2.iloc[:, 0]).mean()-mean1*mean2)/(std1*std2)

    return corr

#print(corr())

score, pvalue, _ = coint(stockdata[['AAPL']], stockdata[['MSFT']])
correlation = corr()

print('Correlation between %s and %s is %f' % ('AAPL', 'MSFT', correlation))
print('Cointegration between %s and %s is %f' % ('AAPL', 'MSFT', pvalue))

plt.figure(figsize=(12,5))
ax1 = df1.iloc[1325: , 11].plot(color='green',grid=True, label='MSFT')
ax2 = df2.iloc[: , 11].plot(color='red',grid=True, secondary_y = True, label = 'AAPL')
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
plt.legend(h1+h2, l1+l2, loc=2)
#plt.show()


#TODO: returnHedgeRatio = beta via OLS regression, statistical thingies (z-score, volatility, mean, stddev)
#      also shareHedgeRatio = (PxA/PxB)*retHedgeRatio
#      number of dollars traded, average time in trade, annualized ROI

#output = pd.concat([df1[['Date', 'aClose']][df1.index>=1325],df2[['aClose']],Spr,Corr], axis = 1)
#print(Spr)

output = pd.concat([stockdata,Spr,Ret], axis = 1)
#print(output)
output.to_excel(r'/Users/piersonmichalak/Desktop/blankpair.xlsx',sheet_name ='sheet1',
              index = True)