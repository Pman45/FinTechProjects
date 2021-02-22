import quandl
import pandas as pd
from matplotlib import pyplot as plt
import requests
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.stattools import coint
import yfinance as yf
from yahoofinancials import YahooFinancials
import numpy as np

# assets = ['TSLA', 'MSFT', 'FB']
#
# yahoo_financials = YahooFinancials(assets)
#
# data = yahoo_financials.get_historical_price_data(start_date='2019-01-01',
#                                                  end_date='2019-12-31',
#                                                  time_interval='weekly')
#
# prices_df = pd.DataFrame({
#     a: {x['formatted_date']: x['adjclose'] for x in data[a]['prices']} for a in assets
# })


def findduplicates(nums):
    tortoise = nums[0]
    hare = nums[0]


    while True:
        tortoise=nums[tortoise]
        hare = nums[nums[hare]]
        if tortoise == hare:
            break

    ptr1 = nums[0]
    ptr2 = tortoise
    while ptr1!=ptr2:
        ptr1=nums[ptr1]
        ptr2=nums[ptr2]

    return ptr1

#print(findduplicates([3,1,5,4,2,6,7,8,9,9]))

def createdf():
    test = pd.DataFrame()
    titles = ['A','B','C','D']
    headers = ['a','b','c']
    dfs = []
    dd = {} #this makes a dictionary
    for i in range(6):
        name = 'df' + str(i)
        dd[name] = pd.DataFrame(np.random.randint(0,100,size=(10, 2)), columns=list('AB'))
        #print(dd[name])
        #output = dd[name].append(test)
        dfs.append(dd[name])

    output = pd.concat(dfs, axis=1)
    df_multiindex = pd.MultiIndex.from_product(
        (#('Summer', 'Autumn', 'Winter', 'Spring'),
            (titles),
         (headers))
    )
    #output.columns = df_multiindex
    #output.columns = list
    print(output)
    return

print(createdf())

def inputtest():
    n = int(input("How many stocks would you would like to investigate "))
    dd = {}
    tickers = []
    for i in range(1, n + 1):
        name  = 'a' + str(i)
        name = str(input("Stock" + str(i) + "Ticker: "))
        tickers.append(name)

    print(tickers)
    return


import matplotlib.pyplot as plt
import openpyxl
# Your plot generation code here...
Xreturns = np.random.normal(0,1,100)
X = pd.Series(np.cumsum(Xreturns), name = 'X') +50
Yreturns = np.random.normal(0,1,100)
Y = pd.Series(np.cumsum(Yreturns), name = 'Y') + 50
Z = X.rolling(10).corr(Y)
print(Z)
# X.plot(figsize=(15,7))
# plt.savefig("myplot.png", dpi = 150)
#
# wb = openpyxl.load_workbook(r'/Users/piersonmichalak/Desktop/input.xlsx')
# ws = wb.active
#
# img = openpyxl.drawing.image.Image('myplot.png')
#
#
# ws.add_image(img)
# wb.save(r'/Users/piersonmichalak/Desktop/output.xlsx')