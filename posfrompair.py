from tiingo import TiingoClient
import pandas as pd
import numpy as np
import scipy.stats as sci
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import openpyxl
import datetime
from sklearn import linear_model
import io

#part1 setup API
config = {}
config['session'] = True
config['api_key'] = "92d790d9ad4704559ef11ce7c8fe9f9810b355ea"
client = TiingoClient(config)

#part2 get the tickers, but not too many, just a few, like 10
#p.s make sure this works before you have them do all of this nonsense,
#cus inputting the tickers manually each time you want to run it is going to be tedious af

#for our sake some good ones are REG vs FRT, KRC vs DEI, AMT vs PLD
def inputtest():
    n = int(input("How many stocks would you would like to investigate "))
    dd = {}

    for i in range(1, n + 1):
        name  = 'a' + str(i)
        name = str(input("Stock" + str(i) + "Ticker: "))
        tickers.append(name)

    return tickers
#inputtest()
tickers =['XLF','VFH','EUFN','FNCL','FXO','UYG','IXG','RYF',
           'GREK','DFNL','SKF','PFI','JHMF']
print(tickers)
#this asks for the tickers so dont trip, next we're getting the top
# cointegrated pairs to test the strategy on
def findcointpairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    '''we need this for the result from coint 
    since coint gives two values '''
    pvalue_matrix = np.ones((n,n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1,n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1,S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i,j] = score
            pvalue_matrix[i,j] = pvalue
            if pvalue < 0.01:
                pairs.append((keys[i], keys[j]))
        return score_matrix, pvalue_matrix, pairs

ticker_px = client.get_dataframe(tickers,
                                frequency='daily',
                                metric_name='adjClose',
                                startDate='2018-01-01',
                                endDate='2020-07-29')
ticker_px.columns = tickers
tickers = ticker_px.columns
print(ticker_px, tickers)
#scores, pvalues, pairs = findcointpairs(ticker_px)
''' 
STRAT 1 Formula: so based on ratios
DesPos A = Ratio-MA where MA is 16 day moving average 
DesPos B = -despos A * ratio 
'''
def distancedata(ratios,px1,px2,Ratio,a1,a2):
    labels = ['DesPos ' + a1, 'DesPos ' + a2,'Ratio'
        , 'ratio_MA', 'ratio - MA', 'Froward Ret_ratio','Ret' + a1, 'Ret' + a2, 'Corr'
        ,'PNL' + a1, 'PNL' + a2  ]
    ratio_MA = ratios.rolling(window = 16, center = False).mean()
    Ratio_MA = pd.DataFrame(ratio_MA)
    Ratio_MA.reset_index(drop= True, inplace = True)
    ratio_minus_MA = pd.DataFrame(ratios - ratio_MA)
    ratio_minus_MA.reset_index(drop= True, inplace = True)
    forwardret_ratio = Ratio.pct_change(periods=7)
    Forward = pd.DataFrame(forwardret_ratio)
    Forward.reset_index(drop= True, inplace = True)
    # Ratiostuff = pd.concat([Ratio_MA,ratio_minus_MA,Forward],axis = 1)
    # Ratiostuff.reset_index(drop = True, inplace = True)
    #Ratiostuff.columns = ['ratio_MA', 'ratio - MA', 'Froward Ret_ratio']
    reta1 = px1.pct_change()#periods = 1)
    Reta1 = pd.DataFrame(reta1)
    reta2 = px2.pct_change()#periods = 1)
    Reta2 = pd.DataFrame(reta2)
    Reta2.reset_index(drop=True, inplace = True)
    retcorr = reta1.rolling(4).corr(reta2)
    Retcorr = pd.DataFrame(retcorr)
    Retcorr.reset_index(drop=True, inplace = True)
    #Rets = pd.DataFrame(Reta1,Reta2)
    Reta1.reset_index(drop = True, inplace = True)
    #Rets.columns = ['Ret' + a1, 'Ret' + a2, 'Corr']
    desposa1 = -1*(ratios-ratio_MA)
    desposa2 = -1*(desposa1)*ratios
    DesPosa1 = pd.DataFrame(desposa1)
    DesPosa1.reset_index(drop = True, inplace = True)
    DesPosa2 = pd.DataFrame(desposa2)
    DesPosa2.reset_index(drop=True, inplace=True)
    #DesPos.columns = ['DesPos ' + a1, 'DesPos ' + a2]
    pnla1 = pd.DataFrame(desposa1*(px1[:-1].values/px1[1:]-1))
    pnla1.reset_index(drop = True, inplace = True)
    pnla2 = pd.DataFrame(desposa2*(px2[:-1].values/px2[1:]-1))
    pnla2.reset_index(drop = True, inplace = True)
    # totalpnl = pnla1[1:]+pnla2[1:]
    # cumpnl = totalpnl.cumsum()
    Pnlstuff = pd.concat([pnla1,pnla2],axis = 1) #,totalpnl,cumpnl)
    Pnlstuff.reset_index(drop=True, inplace = True)
    #Pnlstuff.columns = ['PNL' + a1, 'PNL' + a2, 'total PNL', 'Cum PNL']
    df2 = pd.concat([DesPosa1,DesPosa2,Ratio,Ratio_MA,ratio_minus_MA,Forward,Reta1,Reta2,Retcorr,pnla1,pnla2], axis = 1)
    df2.columns = labels
    return df2
# dfs=[]
# dd={}
# a1 = tickers[0]
# a2 = tickers[1]
# px1 = ticker_px[a1]
# px1.reset_index(drop=True, inplace=True)
# px2 = ticker_px[a2]
# px2.reset_index(drop=True, inplace=True)
# df1 = pd.concat([px1,px2],axis = 1)
# name = a1 + 'vs' + a2
# print(a1 + ' and ' + a2 + ' in progress')
# ratios = px1 / px2
# Ratio = pd.DataFrame(ratios)
# Ratio.columns = ['Ratio' + a1 + 'vs' + a2]
# Ratio.reset_index(drop=True, inplace=True)
# df2 = distancedata(ratios,px1,px2,Ratio,a1,a2)
# dd[name] = pd.concat([df1, df2], axis=1)
# dfs.append(dd[name])
# print(dfs)
''' 
STRAT 2 Formula 
'''
def corrbasedata():
    return
'''
STRAT 3 Formula 
'''
def cointbasedata():
    return

def autopairdata():
    n = ticker_px.shape[0]
    # pairpx = np.zeros((n,161), np.float64) #Idk the size of this yet, looking kinda slim rn ngl
    #labels = []
    # pairheaders = ['Px1', 'Px2', 'Ret Corr', 'Spread', 'Beta Spread', 'SpreadMA']
    dfs = []
    dd = {}
    for a1 in ticker_px.columns:
        for a2 in ticker_px.columns:
            if a1 != a2:
                # dd = {}
                test_result = ts.coint(ticker_px[a1], ticker_px[a2])
                if test_result[1] < 0.07:
                    print(a1,a2)
                    # dfs = [] <- if you define this each time, it resets dfs to the empty list, so dont do that dumbass
                    #labels.append(([a1], [a2]))
                    px1 = ticker_px[a1]
                    px1.reset_index(drop=True, inplace=True)
                    px2 = ticker_px[a2]
                    px2.reset_index(drop=True, inplace=True)
                    df1 = pd.concat([px1,px2],axis = 1)
                    name = a1 + 'vs' + a2
                    print(a1 + ' and ' + a2 + ' in progress')
                    ratios = px1 / px2
                    Ratio = pd.DataFrame(ratios)
                    Ratio.columns = ['Ratio' + a1 + 'vs' + a2]
                    Ratio.reset_index(drop=True, inplace=True)
                    df2 = distancedata(ratios,px1,px2,Ratio,a1,a2)
                    dd[name] = pd.concat([df1, df2], axis=1)
                    dfs.append(dd[name])

    pairpx = pd.concat(dfs,axis = 1)
    return pairpx
# dfs = list()
# dd = {}
# for a1 in ticker_px.columns:
#     for a2 in ticker_px.columns:
#          if a1 != a2:
#                 # dd = {}
#             test_result = ts.coint(ticker_px[a1], ticker_px[a2])
#             if test_result[1] < 0.09:
#                 print(a1,a2)
#                 # dfs = [] <- if you define this each time, it resets dfs to the empty list, so dont do that dumbass
#                 #labels.append(([a1], [a2]))
#                 px1 = ticker_px[a1]
#                 px1.reset_index(drop=True, inplace=True)
#                 px2 = ticker_px[a2]
#                 px2.reset_index(drop=True, inplace=True)
#                 df1 = pd.concat([px1,px2],axis = 1)
#                 name = a1 + 'vs' + a2
#                 print(a1 + ' and ' + a2 + ' in progress')
#                 ratios = px1 / px2
#                 Ratio = pd.DataFrame(ratios)
#                 Ratio.columns = ['Ratio' + a1 + 'vs' + a2]
#                 Ratio.reset_index(drop=True, inplace=True)
#                 df2 = distancedata(ratios,px1,px2,Ratio,a1,a2)
#                 dd[name] = pd.concat([df1, df2], axis=1)
#                 print(dd[name])
#                 dfs.append(dd[name])
# print(dfs)
#pairpx = pd.concat(dfs,axis = 1)

pairpx = autopairdata()
print(pairpx)
pairpx.to_excel(r'/Users/piersonmichalak/Desktop/BFINpairs.xlsx',sheet_name ='sheet1',index = True)