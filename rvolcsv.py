import pandas as pd
import math
import numpy as np
import scipy.stats as sci
import sys
from openpyxl import load_workbook
#import xlwt
#from xlwt import Workbook

header_names = ['Date','rOpen', 'rHigh', 'rLow','rClose', 'rVolume'
    , 'flag1', 'flag2', 'aOpen', 'aHigh', 'aLow', 'aClose', 'aVolume']
df = pd.read_csv('/Users/piersonmichalak/Desktop/spy.csv', header = None, skiprows = 1, names = header_names )

df.reset_index(drop=True, inplace=True)


#print(df.index)
#max is 6887, min is 4518
def retdata():
    ret = np.zeros((2369, 1), np.float64)
    ret[0] = 0
    j = 4517
    k = (252 / 30)
    for t in range(ret.shape[0]):
        # for i in range(30):
        ret[t] = (float(np.log(df.iat[t + j, 11] / df.iat[t + j-1, 11])))#** 2

    return ret
def ret2data():
    ret2 = np.zeros((2368, 1), np.float64)
    ret2[0] = 0
    j = 4517
    k = (252 / 30)
    for t in range(ret2.shape[0]):
        # for i in range(30):
        ret2[t] = (float(np.log(df.iat[t + j, 11] / df.iat[t + j-1, 11])))** 2

    return ret2
np.random.seed(123)
ret = retdata()
ret[:, 0].round(4)
ret2 = ret2data()
ret2[:, 0].round(4)
Ret2 = pd.DataFrame({'Ret^2':ret2[:,0]})
RetV = pd.DataFrame({'Ret':ret[:, 0]})

def rvoldata():
    ret2 = np.zeros((2369, 1), np.float64)
    ret2[0] = 0
    rvol = np.zeros((2369, 1), np.float64)
    rvol[0]= 0
    j = 4517
    k = (252/30)
    for t in range(ret2.shape[0]):
        #for i in range(30):
        ret2[t] = (float(np.log(df.iat[t + j, 11] / df.iat[t + j-1, 11])))** 2
    #return ret
    for i in range(2340):

        rvol[i+29] = np.sqrt(k*np.sum(ret2[i:28+i]))
        #print(len(ret2[i:28+i]))
        #print('here', 100*np.sqrt((252/30)*np.sum(ret2[i:30+i])))

        #print('here', 100* np.sqrt(k*np.sum((ret[30*(i):30*(i+1)+1])**2)) )

    return rvol
        #df.iloc[t-30:t, 11]

np.random.seed(123)
rvol = rvoldata()
rvol[:, 0].round(10)
#print rvol
#print(rvol.shape)
#print ret


orig = df[['Date','aClose']][df.index >= 4516 ]
orig.reset_index(drop = True, inplace = True)

#d={'rVol':[rvol[:,]]}
#d = np.array([rvol[:, ]])
#rVol = pd.DataFrame(data=d, columns = ['rVol'])

rVol = pd.DataFrame({'rVol':rvol[:, 0]})
#rVol.pivot(columns='rVol', values='rvol')
orig = orig.append(dict(zip(orig.columns, rvol)), ignore_index=True)
#orig.insert(2, 'rVol',rvol, True)
output = pd.concat([orig,rVol,RetV,Ret2], axis = 1)
#print(Ret2)
#print(RetV)
print(rVol)
#print(orig)
#print(orig.shape)
#print(output)

book = load_workbook(r'/Users/piersonmichalak/Desktop/optionProject1_example.xlsx')

output.to_excel(r'/Users/piersonmichalak/Desktop/test.xlsx',sheet_name ='datastuff',
             startrow=6, startcol = 3, index = True)
#output.to_excel(r'/Users/piersonmichalak/Desktop/optionProject1_example.xlsx',sheet_name ='datastuff',
             #startrow=6, startcol = 3, index = False, header = None   )

#df.set_index('Date',inplace = True)
#rows, columns = df.shape
#print(df[2:5])
#print(df.head())
#print(df.columns)
#print(df['aClose'].max())
#df = df[~df.index.duplicated()]

#df.set_index(range(6886),inplace = True)