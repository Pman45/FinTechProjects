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
config = {}
config['session'] = True
config['api_key'] = "92d790d9ad4704559ef11ce7c8fe9f9810b355ea"
client = TiingoClient(config)
tickers = []
def inputtest():
    n = int(input("How many stocks would you would like to investigate "))
    dd = {}

    for i in range(1, n + 1):
        name  = 'a' + str(i)
        name = str(input("Stock" + str(i) + "Ticker: "))
        tickers.append(name)

    return tickers
inputtest()
print(tickers)
startDate = '2016-01-01'
endDate = '2020-07-21'
ticker_px = client.get_dataframe(tickers,
                                frequency='daily',
                                metric_name='adjClose',
                                startDate='2016-01-01',
                                endDate='2020-07-21')
ticker_px.index.tz_convert(None)
ticker_px.tz_localize(None)
ticker_px.reset_index(inplace = True, drop = True)
print(ticker_px)

def cointpairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n,n))
    pvalue_matrix = np.ones((n,n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.008:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs

class pairinfo(object):
    def __init__(self, a1, a2):
        self.a1 = a1
        self.a2 = a2
        self.pxa1 = px1
        self.pxa2 = px2
#musth ave ratio, ratioMA, ratio-MA, forwardReRatio rets, cumrets correlation and cointegration constants (coint is just corr on forward ret pair ratio) )
#then for the deep backtest we need Dail PNL A, B total pNL, CUM PNL, and desired positions on each asset
class backtestinfo(object):
    def __init__(self, a1, a2):
        self.a1 = a1
        self.a2 = a2
        px1 = ticker_px[a1]
        self.px1 = px1
        px2 = ticker_px[a2]
        self.px2 = px2
        ratios = ticker_px[a1] / ticker_px[a2]
        self.ratios = ratios


    def ratiodata(self, a1,a2):
        ratios = ticker_px[a1] / ticker_px[a2]
        return ratios

    def zscoredata(self,a1,a2):
        zscore = (ratios - ratios.mean()) / np.std(ratios)
        return zscore

    def movingdata(self,px1,px2,ratios):
        ratios_ma5 = ratios.rolling(window = 5,center =False).mean()
        ratios_ma20 = ratios.rolling(window = 20, center = False).mean()
        ratios_ma60 = ratios.rolling(window = 60, center = False).mean()
        std60 = ratios.rolling(window=60, center = False).std()
        zscore_60_5 = (ratios_ma5 - ratios_ma60)/std60
        ratio_minus_ma = ratios - ratios_ma20

        df = pd.DataFrame([ratios_ma5, ratios_ma20, ratios_ma60, std60, zscore_60_5, ratio_minus_ma])
        Ratiodata=df.T
        return Ratiodata

    def retdata(self,px1,px2,ratios):
        n = ticker_px.shape[0]
        forwardret = np.zeros((n, 1),  np.float64)
        px1rets = np.zeros((n, 1), np.float64)
        px2rets = np.zeros((n, 1), np.float64)
        forwardret[0][0] = np.nan
        for i in range(20):
            forwardret[i][0] = np.nan
        px1rets[0][0] = np.nan
        px2rets[0][0] = np.nan
        for i in range(21,ticker_px.shape[0]):
            forwardret[i][0] = ratios[i]/ratios[i-20] - 1

        for i in range(1,ticker_px.shape[0]):
            px1rets[i][0] = px1[i]/px1[i-1]-1
            px2rets[i][0] = px2[i]/px2[i-1]-1

        return forwardret,px1rets, px2rets


#outofclasssnow, so we need to get a dataframe to make the position stuff
a1 = tickers[0]
a2 = tickers[1]
pair = backtestinfo(a1,a2)
ratios = pair.ratios
px1 = ticker_px[a1]
px2 = ticker_px[a2]
#print(pair.movingdata(px1,px2,ratios))

#print(a1,a2)
#get the obligatory df stuff out the way
def dfdata():
    ratios = pair.ratios
    px1 = ticker_px[a1]
    px2 = ticker_px[a2]
    Px1 = pd.DataFrame(px1)
    Px2 = pd.DataFrame(px2)
    # Px1.reset_index(drop = True, inplace = True)
    # Px2.reset_index(drop=True, inplace=True)
    #Px1.index = pd.bdate_range(startDate, endDate, freq='B')
    #Px2.index = pd.bdate_range(startDate, endDate, freq ='B')
    Ratiodf = pd.DataFrame(pair.ratios)
    Movingdf = pd.DataFrame(pair.movingdata(px1,px2,ratios))
    df = pd.concat([Px1, Px2, Ratiodf, Movingdf], axis = 1)
    #df.index.tz_convert(None)
    #print(df.index())
    headers = [a1, a2, 'Ratios '+ a1 + '/' + a2, 'ratio ma_5', 'ratio ma_20', 'ratio ma_60', 'ratio ma_60 std', 'zscore_ma60_vs_ma5', 'ratio minus ma']
    df.columns = headers
    #Rets = pd.DataFrame(pair.retdata(px1,px2,ratios))
    return df #Rets
#TODO: fix the returns function, somethings bugging

def plotdata():

    ratios = pair.ratios
    buy = ratios.copy()
    sell = ratios.copy()
    zscore = pair.zscoredata(a1,a2)
    ratios_ma5 = ratios.rolling(window=5, center=False).mean()
    ratios_ma60 = ratios.rolling(window=60, center=False).mean()
    std60 = ratios.rolling(window=60, center=False).std()
    zscore_60_5 = (ratios_ma5 - ratios_ma60) / std60
    buy[zscore_60_5> -1 ] = 0
    sell[zscore_60_5<1] = 0
    ratios[60:].plot()
    ratios_ma5[60:].plot()
    buy[60:].plot(color='g', linestyle ='None', marker = 6)

    sell[60:].plot(color='r', linestyle ='None', marker =6)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, ratios.min(), ratios.max()))
    plt.legend(['Ratio', 'Ratios_ma5','BuySignal', 'SellSignal'])
    #plt.show()
    plt.savefig("ratioplot.png", dpi=150)
    img = openpyxl.drawing.image.Image('ratioplot.png')
    wb = openpyxl.load_workbook(r'/Users/piersonmichalak/Desktop/input.xlsx')
    ws = wb.active
    ws.add_image(img)
    wb.save(r'/Users/piersonmichalak/Desktop/output2.xlsx')
    buyR = 0*px1.copy()
    sellR = 0*px1.copy()

    # When buying the ratio, buy S1 and sell S2
    buyR[buy != 0] = px1[buy != 0]
    sellR[buy != 0] = px2[buy != 0]
    # When selling the ratio, sell S1 and buy S2
    buyR[sell != 0] = px2[sell != 0]
    sellR[sell != 0] = px1[sell != 0]

    px1[60:].plot(color = 'b')
    px2[60:].plot(color = 'c')
    buyR[60:].plot(color='g', linestyle='None', marker=6)
    sellR[60:].plot(color='r', linestyle='None', marker=6)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, min(px1.min(), px2.min()), max(px1.max(), px2.max())))
    plt.legend([a1, a2, 'Buy Signal', 'Sell Signal'])
    #plt.show()
    plt.savefig("buysellplot.png", dpi=150)
    img = openpyxl.drawing.image.Image('buysellplot.png')
    wb = openpyxl.load_workbook(r'/Users/piersonmichalak/Desktop/input.xlsx')
    ws = wb.active
    ws.add_image(img)
    wb.save(r'/Users/piersonmichalak/Desktop/output3.xlsx')
    return
#plotdata()
def desposdata(px1,px2, window1,window2):
    if (window1==0) or (window2==0):
        return 0

    ratios = pair.ratios
    ma1 = ratios.rolling(window = window1, center = False).mean()
    ma2 = ratios.rolling(window=window2, center = False).mean()
    std = ratios.rolling(window = window2, center = False).std()
    zscore = (ma1-ma2)/std
    cash = 0
    countpx1 = 0
    countpx2 = 0
    roi = np.zeros((ticker_px.shape[0], 1), np.float64)
    roi[0][0] = 0
    despos = np.zeros((ticker_px.shape[0], 2), np.float64)
    despos[0][0] = 0
    despos[0][1] = 0
    for i in range(1, ticker_px.shape[0]):
        if zscore[i] > 1: #sell short if zscore too big (denom growing)
            cash += px1[i] - px2[i]*ratios[i]
            roi[i][0] += cash
            despos[i][0] = despos[i-1][0]- 1 #b4 we used despos[i][0]=-1 despos[i][1]=1
            despos[i][1] = ratios[i] + despos[i-1][1]
            countpx1 -= 1
            countpx2 += ratios[i]
        elif zscore[i] < -1:
            cash -= px1[i] - px2[i]*ratios[i]
            roi[i][0] += cash
            despos[i][0] = ratios[i]+despos[i-1][0]#b4 we used despos[i][0] = 1 despos[i][1] = -1
            despos[i][1] = 1 + despos[i-1][1]
            countpx1 += ratios[i]
            countpx2 -= 1

        elif (abs(zscore[i])<0.75):
            cash += px1[i]*countpx1 + px2[i]*countpx2
            roi[i][0] += cash
            despos[i][0] = 0 + despos[i-1][0]
            despos[i][1] = 0 + despos[i-1][1]
            countpx1 = 0
            countpx2 = 0


    Despos = pd.DataFrame(despos)
    Roi = pd.DataFrame(roi)
    df2 = pd.concat([Despos,Roi],axis = 1)
    return df2
window1=5
window2=20
# window1=int(input('What is the desired lookback window for the first moving average: '))
# window2=int(input('What is the desired lookback window for the second moving average: '))

df1 = dfdata()
df2 = desposdata(px1,px2,window1,window2)
df2.columns = ['Pos' + a1, 'Pos' + a2, 'RoI' ]
#print(df2)
output = pd.concat([df1,df2], axis = 1)

output.to_excel(r'/Users/piersonmichalak/Desktop/output.xlsx',sheet_name ='sheet1',index = True)