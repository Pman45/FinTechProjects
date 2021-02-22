import cv2
import pandas as pd
import numpy as np
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf
def add(a,b):
  return a+b

print(add(3,4))

yf.pdr_override() #activates the yfinance workaround

stock = input("Enter a stock ticker smybol here:") #prompts user to input a stock ticker symbol
print(stock)
startyear = 2020
startmonth = 1
startday = 1

start=dt.datetime(startyear,startmonth,startday) #this format sets up the start date for data collection
now= dt.datetime.now() #makes it so that the end date is now, but this can be modified

df=pdr.get_data_yahoo(stock,start,now) #this uses python data reader and uses data from yahoo's stock quotes

print(df)

ma=50 #this makes the moving average 50

smaString = "Sma_"+str(ma) #this is the title of the column for the moving average

df[smaString] = df.iloc[:,4].rolling(window=ma).mean()
#creates column with name smaString, then this is the rolling
#moving average
print(df)


