from tiingo import TiingoClient
import pandas as pd
import numpy as np
import scipy.stats as sci
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import datetime
from sklearn import linear_model
#got this class from quantconnect so let's put it to work
class pairs(object):
    def __init__(self, a,b):
        self.a = a
        self.b = b
        self.name = str(a) + ':' + str(b)
        self.df = pd.concat([a.df,b.df], axis = 1).dropna()
    #next we want to determine the number of bars in the rolling window, which we do
    #by determinig the shape of the DataFrame
        self.num_bar = self.df.shape[0]
        self.cor = self.df.corr().ix[0][1]
    #This sets the initial signals to be 0
        self.error = 0
        self.last_error = 0
        self.a_price = []
        self.a_date = []
        self.b_price = []
        self.b_date = []

    #display correlation matrix
    def cor_update(self):
        self.cor = self.dr.corr().ix[0][1]
    #gets cointegration info
    def cointegration_test(self):
        self.model  = sm.ols(formula= '%s ~ %s'%(str(self.a),str(self.b)), data = self.df).fit()
        self.adf = ts.adfuller(self.model.resid,autolag = 'BIC')[0]
        self.mean_error = np.mean(self.model.resid)
        self.sd = np.std(self.model.resid)

    def price_record(self, data_a, data_b):
        self.a_price.append(float(data_a.Close))
        self.a_date.append(data_a.EndTime)
        self.b_price.append(float(data_b.Close))
        self.b_date.append(data_b.EndTime)

    def df_update(self):
        new_df = pd.DataFrame({str(self.a):self.a_price,str(self.b):self.b_price},index = [self.a_date]).dropna()
        self.df = pd.concat([self.df,new_df])
        self.df = self.df.tail(self.num_bar)
        for i in [self.a_price,self.a_date,self.b_price,self.b_date]:
            i = []
#CSVinfo = str(input("CSV title:"))
#Headers = str(input("Header names: "))

# config = {}
# config['session'] = True
# config['api_key'] = "92d790d9ad4704559ef11ce7c8fe9f9810b355ea"
# client = TiingoClient(config)
# historical_prices = client.get_ticker_price("GOOGL",
#                                             fmt='json',
#                                             startDate='2017-08-01',
#                                             endDate='2017-08-31',
#                                             frequency='daily')
#valid data items are: {'volume', 'divCash', 'adjHigh'
# , 'adjVolume', 'adjLow', 'high', 'adjOpen', 'low', 'adjClose', 'close', 'open', 'splitFactor'}
#'INVH', 'REXR', 'STOR','CBRE', 'COLD', 'JILL', 'VICI'
tickers = ['SPY','AMT', 'CCI', 'PLD', 'EQIX', 'DLR', 'BXP',
            'ESS', 'PEAK', 'VTR', 'WELL', 'CSGP','AVB', 'SUI',
           'DRE', 'MAA', 'EXR', 'WPC', 'MPW', 'ELS', 'NLY',  'CONE', 'CPT', 'IRM', 'HST',
           'REG', 'AGNC', 'OHI', 'GLPI', 'AMH', 'VER', 'KRC', 'NNN', 'HTA', 'VNO', 'FRT', 'AIV',
            'LAMR', 'KIM', 'CUBE', 'COR', 'DEI', 'EGP'
         ]
#print(len(tickers))
# ticker_px = client.get_dataframe(tickers,
#                                 frequency='daily',
#                                 metric_name='adjClose',
#                                 startDate='2016-01-01',
#                                 endDate='2020-07-13')
from backtester.dataSource.yahoo_data_source import YahooStockDataSource
from datetime import datetime
startDateStr = '2016/01/01'
endDateStr = '2020/07/10'
cachedFolderName = 'yahooData/'  # the folder to which the info is downloaded
dataSetId = 'testPairsTrading'  # keep this
instrumentIds = tickers
ds = YahooStockDataSource(cachedFolderName=cachedFolderName,
                            dataSetId=dataSetId,
                            instrumentIds=instrumentIds,
                            startDateStr=startDateStr,
                            endDateStr=endDateStr,
                            event='history')
ticker_px = ds.getBookDataByFeature()['adjClose']
ticker_px.columns = tickers



def pvaluedata():

    pvaluedf = pd.DataFrame()
    n = ticker_px.shape[0]

    for a1 in ticker_px.columns:
        for a2 in ticker_px.columns:
            if a1 != a2:
                test_result = ts.coint(ticker_px[a1], ticker_px[a2])
                if test_result[1] < 0.008:
                    print(a1 + ' and ' + a2 + ': p-value = ' + str(test_result[1]))

    print('These are the top pairs ')

    return

def cointpairs(data):
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
            if pvalue < 0.008:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs

#scores, pvalues, pairs = cointpairs(ticker_px)
#print(pairs)
class bigstuff(object):
    n = ticker_px.shape[0]
    bigstuff = np.zeros((n, 23), np.float64)
    def __init__(self, a1, a2):
        self.a1 = a1
        self.a2 = a2
        self.name = str(a1) + ':' + str(a2)
        #self.df = pd.concat([a1, a2], axis=1).dropna()
        #self.cor = self.df.corr().ix[0][1]
        px1 = pd.DataFrame([ticker_px[a1]])
        px2 = pd.DataFrame([ticker_px[a2]])
        self.a1px = px1
        self.a2px = px2



    # display correlation matrix
    # def cor_update(self):
    #     self.cor = self.dr.corr().ix[0][1]

    def corrret(self, px1,px2):
        n = ticker_px.shape[0]
        Ret = np.zeros((n,2), np.float64)
        Ret[0][0] = 0
        Ret[0][1] = 0
        for i in range(ticker_px.shape[0]):
            Ret[i][0] = (px1.iat[i] / px1.iat[i - 1] - 1) * 100
            Ret[i][1] = (px2.iat[i] / px2.iat[i - 1] - 1) * 100

        Rets = pd.DataFrame(Ret)
        corr = Rets.corr()
        return corr
    def reg(self, px1, px2):
        regr = linear_model.LinearRegression()
        a1_constant = pd.concat([px1, pd.Series([1] * len(px1), index=px1.index)], axis=1)
        regr.fit(a1_constant, px2)
        beta = regr.coef_[0]
        alpha = regr.intercept_
        spread = px2 - px1 * beta - alpha
        return spread


    def betaspreaddata(self, a1,a2, px1, px2):
        def beta(self, px1, px2):
            regr = linear_model.LinearRegression()
            a1_constant = pd.concat([px1, pd.Series([1] * len(px1), index=px1.index)], axis=1)
            regr.fit(a1_constant, px2)
            beta = regr.coef_[0]
            alpha = regr.intercept_
            #spread = px2 - px1 * beta - alpha
            return beta

        def adjretdata(self, px1, px2):
            adjret = np.zeros((ticker_px.shape[0], 2), np.float64)
            adjret[0][0] = (px1.iat[0]) / (px1.iat[0])
            adjret[0][1] = (px1.iat[0]) / (px1.iat[0])
            for i in range(ticker_px.shape[0]-1):
                adjret[i][0] = (px1.iat[i] / px1.iat[i - 1] - 1) * 100  # return1
                adjret[i][1] = (px2.iat[i] / px2.iat[i - 1] - 1) * 100  # return2
            return adjret

        np.random.seed(123)
        beta = beta(self, px1,px2)
        adjret = adjretdata(self, px1, px2)
        Rets = pd.DataFrame({'AdjRet1': adjret[:, 0], 'AdjRet2': adjret[:, 1]})
        CumRet = Rets.cumsum()
        CumRet.columns = ['CumRet1', 'CumRet2']
        betaspread = np.zeros((ticker_px.shape[0],1), np.float64)
        betaspread[0][0]= CumRet.iat[0,0] - beta*CumRet.iat[0,1]
        for j in range(ticker_px.shape[0]):
            betaspread[j][0] = CumRet.iat[j, 0] - beta*CumRet.iat[j, 1]

        BetaSpread = pd.DataFrame(betaspread)
        SMA = np.zeros((BetaSpread.shape[1], 1), np.float64)
        SMA = BetaSpread.rolling(20).mean()
        return BetaSpread  #calculate beta cum ret spread
    def SMAdata(self, px1,px2):
        def beta(self, px1, px2):
            regr = linear_model.LinearRegression()
            a1_constant = pd.concat([px1, pd.Series([1] * len(px1), index=px1.index)], axis=1)
            regr.fit(a1_constant, px2)
            beta = regr.coef_[0]
            alpha = regr.intercept_
            # spread = px2 - px1 * beta - alpha
            return beta

        def adjretdata(self, px1, px2):
            adjret = np.zeros((ticker_px.shape[0], 2), np.float64)
            adjret[0][0] = (px1.iat[0]) / (px1.iat[0])
            adjret[0][1] = (px1.iat[0]) / (px1.iat[0])
            for i in range(ticker_px.shape[0] - 1):
                adjret[i][0] = (px1.iat[i] / px1.iat[i - 1] - 1) * 100  # return1
                adjret[i][1] = (px2.iat[i] / px2.iat[i - 1] - 1) * 100  # return2
            return adjret

        np.random.seed(123)
        beta = beta(self, px1, px2)
        adjret = adjretdata(self, px1, px2)
        Rets = pd.DataFrame({'AdjRet1': adjret[:, 0], 'AdjRet2': adjret[:, 1]})
        CumRet = Rets.cumsum()
        CumRet.columns = ['CumRet1', 'CumRet2']
        betaspread = np.zeros((ticker_px.shape[0], 1), np.float64)
        betaspread[0][0] = CumRet.iat[0, 0] - beta * CumRet.iat[0, 1]
        for j in range(ticker_px.shape[0]):
            betaspread[j][0] = CumRet.iat[j, 0] - beta * CumRet.iat[j, 1]

        BetaSpread = pd.DataFrame(betaspread)
        SMA = np.zeros((BetaSpread.shape[1], 1), np.float64)
        SMA = BetaSpread.rolling(20).mean()
        return SMA


def pairpx_data():
    #panels = {i: ticker_px[i] for i in pairs}
    # pairpx = pd.concat([panels[i]['Adj Close'] for i in pairs], axis=1
    #                , names=my_tickers, ignore_index=True)
    #pairpx = pd.concat([ticker_px[i] for i in pairs], axis = 1, ignore_index = True)
    n = ticker_px.shape[0]
    #pairpx = np.zeros((n,161), np.float64) #Idk the size of this yet, looking kinda slim rn ngl
    labels = []
    #pairheaders = ['Px1', 'Px2', 'Ret Corr', 'Spread', 'Beta Spread', 'SpreadMA']
    dfs = []
    dd = {}
    for a1 in ticker_px.columns:
        for a2 in ticker_px.columns:
            if a1 != a2:
                #dd = {}
                test_result = ts.coint(ticker_px[a1], ticker_px[a2])
                if test_result[1] < 0.008:
                    #dfs = [] <- if you define this each time, it resets dfs to the empty list, so dont do that dumbass
                    labels.append(([a1],[a2]))
                    px1 = ticker_px[a1]
                    px2 = ticker_px[a2]

                    # we need stock A, stock B, correlation of returns, spread, spreadMA, spread-MA
                    name = a1 + 'vs' + a2
                    pairheaders = [a1, a2, a1 + ' vs ' + a2 + 'Spread', a1 + ' vs ' + a2 + 'Corr', a1 + ' vs ' + a2 + 'Beta Spread', a1 + ' vs ' + a2 + 'SpreadMA','Spread-MA']
                    pair = bigstuff(a1,a2)
                    #print(pair.corrret(px1,px2).min()[0])
                    print(a1 + ' and ' +  a2  + ' in progress')
                    corrarray = np.zeros((ticker_px.shape[0]),np.float64)
                    corrarray[0]=0
                    for i in range(ticker_px.shape[0]):
                        corrarray[i] = pair.corrret(px1,px2).min()[0]

                    CorrF = pd.DataFrame(corrarray)
                    Px1 = pd.DataFrame(px1)
                    Px2 = pd.DataFrame(px2)
                    Spread = pd.DataFrame(pair.reg(px1, px2))

                    BetaSpread = pd.DataFrame(pair.betaspreaddata(a1, a2, px1, px2))
                    SMAspread = pd.DataFrame(pair.SMAdata(px1,px2))
                    spread_minus_ma = pair.betaspreaddata(a1,a2,px1,px2)-pair.SMAdata(px1,px2)
                    Diff = pd.DataFrame(spread_minus_ma)
                    test1df = pd.concat([CorrF, SMAspread, BetaSpread,Diff], axis=1)
                    test1df.index = ticker_px.index
                    testdf = pd.concat([ Px1, Px2, Spread], axis = 1)
                    test2df = testdf[testdf.index >= '2016-01-01']
                    #dd[name] = pd.DataFrame([Px1, Px2, CorrF, pair.reg(px1, px2),pair.betaspreaddata(a1, a2, px1, px2), pair.SMAdata(px1,px2)])
                    dd[name] = pd.concat([test2df,test1df],axis=1)
                    # dd[name] = pd.DataFrame({a1:[px1], a2:[px2], 'Ret Corr':[CorrF],
                    #                          'Spread':[pair.reg(px1,px2)] , 'Beta Spread':[pair.betaspreaddata(a1,a2,px1,px2)], 'SpreadMA':[pair.SMAdata(px1,px2)]})
                    dd[name].columns = pairheaders
                    dfs.append(dd[name])
                    #print(dd[name])
                    #print(dfs)
    print(labels)
    pairpx = pd.concat(dfs, axis = 1)

    # df_multiindex = pd.MultiIndex.from_product(
    #      ((labels),
    #       (pairheaders)))
    # pairpx.columns = df_multiindex
    #pairpx.columns = labels
    #print(type(labels))
    #print(pairpx)
    return pairpx


pairpx = pairpx_data()
print(pairpx_data())
#print(pvaluedata())
#TODO:find 25 most cointegrated pairs, then for every pair, get CorrReturns,spread-MA(spread)


#print(ticker_px.info())
#pairpx.to_excel(r'/Users/piersonmichalak/Desktop/blankaf.xlsx',sheet_name ='sheet1',index = True)
