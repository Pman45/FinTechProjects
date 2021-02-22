import pandas as pd
import scipy.stats as stats
import numpy as np
import math
import multiprocessing as mp
import datetime
from datetime import timedelta
from openpyxl import load_workbook
import sys

def congru(a, b, m):
    if b == 0:
        return 0

    if a < 0:
        a = -a
        b = -b

    b %= m
    while a > m:
        a -= m

    return (m * congru(m, -b, a) + b) // a


book = load_workbook(r'/Users/piersonmichalak/Desktop/test.xlsx')
headnames = ['Time Step', 'Date', 'aClose', 'rVol', 'iVol', 'DaysTilExp','YearsTilExp', 'Ret','Ret^2']
df = pd.read_excel('/Users/piersonmichalak/Desktop/test.xlsx', 'datastuff',header = None,
                  names = headnames,skiprows = 7, index_col=None )
#df.reset_index(drop = True, inplace = True)
#df.set_index('Time Step')

#print(df)
M=30
r=0
S0 = df.iat[0,2]
K0 = df.iat[0,2]
tau0 = df.iat[0,6]
sigma0 = df.iat[0,4]
n1 = (np.log(S0/K0) + (r + 0.5*sigma0**2)*tau0)/(sigma0 * np.sqrt(tau0))
n2 =(np.log(S0/K0) + (r-0.5*sigma0**2)*tau0)/(sigma0 * np.sqrt(tau0)) #n1-sigma0*np.sqrt(tau0)#
q = 0  # q is the dividend risk rate

delta = np.exp(-q * tau0) * stats.norm.cdf(n1, 0.0, 1.0)
gamma = stats.norm.pdf(n1)/(S0*sigma0*tau0**0.5)
theta = ((-S0*stats.norm.pdf(n1)*sigma0)/(2*np.sqrt(tau0))-r*K0*np.exp(-r*tau0)*stats.norm.cdf(-n1,0.0,1.0))/252
vega = S0*stats.norm.pdf(n1,0.0,1.0)*np.sqrt(tau0)/100
rho = K0*tau0*np.exp(-r*tau0)*stats.norm.cdf(n2)/100
print(gamma,theta,vega)
print(float(252/30))
def greeksdata():

    r=0
    greeks = np.zeros((2370,8), np.float64 )
    greeks[0][0] = S0*stats.norm.cdf(n1,0.0,1.0) - K0*np.exp(-r*tau0)*stats.norm.cdf(n2,0.0,1.0)
    greeks[0][1]= delta#eurocalloption._init_.
    greeks[0][2]= gamma#eurocalloption._init_.self.
    greeks[0][3]= theta#eurocalloption._init_.self.
    greeks[0][4]= rho#eurocalloption._init_.self.
    greeks[0][5]= vega#eurocalloption._init_.self.
    greeks[0][6] = 0 #PNL
    greeks[0][7] = K0 #strike price

    #(np.log(greeks[0][0] / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau)) = n1
    #(np.log(greeks[0][0] / K) - (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau)) = n2
    for i in range(1,2370):
        #dt = float(df.iat[i,5])/30

        tau = df.iat[i, 6]
        if df.iat[i,6] == 0:
            tau = df.iat[i-1,6]

        sigma = df.iat[i,4]

        if i%30 == 0:
            greeks[i][7] = df.iat[i, 2]
        else:
            greeks[i][7] = greeks[i - 1][7]

        #greeks[i][0] = greeks[i - 1][0] * np.exp((r - 0.5 * sigma ** 2) * tau + sigma * np.sqrt(tau) )
        d1 = (np.log(df.iat[i, 2]/ greeks[i][7]) + (0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
        d2 = (np.log(df.iat[i, 2]/ greeks[i][7]) - ( 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))

        greeks[i][0] = (df.iat[i,2]*stats.norm.cdf(d1) - greeks[i][7]*np.exp(-r*tau)*stats.norm.cdf(d2))

        greeks[i][1] = stats.norm.cdf(d1)
        greeks[i][2] = greeks[i][7]*(stats.norm.pdf(d2,0.0,1.0))/(df.iat[i,2]**2 * sigma * np.sqrt(tau))
            #stats.norm.pdf(d1,0.0,1.0)/(df.iat[i,2]*sigma*np.sqrt(tau))
        greeks[i][3] = (-df.iat[i,2]*stats.norm.pdf(-d1,0.0,1.0)*df.iat[i,4])/(2*np.sqrt(tau))/252
                       #-r*greeks[i][7]*np.exp(-r*tau)*stats.norm.cdf(-d1,0.0,1.0)

        greeks[i][4] = greeks[i][7]*tau*np.exp(-r*tau)*stats.norm.cdf(d2,0.0,1.0)
        greeks[i][5] = df.iat[i,2]*stats.norm.pdf(d1,0.0,1.0)*np.sqrt(tau)/100

        if i%30 ==0:
            greeks[i][6] = 0
        else:
            greeks[i][6] = greeks[i][0] - greeks[i-1][0]

        #((df.iat[i,2]*stats.norm.cdf(d1,0.0,1.0) - greeks[i][7]*np.exp(-r*tau)*stats.norm.cdf(d2,0.0,1.0)
                        # - df.iat[i-1,2]*stats.norm.cdf((np.log(df.iat[i-1,2] / greeks[i][7])
                        #+ (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau)),0.0,1.0) +greeks[i][7]*np.exp(-r*tau)*stats.norm
                        # .cdf((np.log(df.iat[i-1,2] / greeks[i][7]) - (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau)),0.0
                         #     ,1.0))) - greeks[i][1]*(df.iat[i,2]-df.iat[i-1,2])


    return greeks


np.random.seed(123)
greeks = greeksdata()
greeks[:, 0].round(4)
greeks[:, 1].round(4)
greeks[:, 2].round(4)
greeks[:, 3].round(4)
greeks[:, 4].round(4)
greeks[:, 7].round(4)

deldf = pd.DataFrame({'Delta':greeks[:,1]})
tgdf = pd.DataFrame({'Gamma':greeks[:, 2],'Theta':greeks[:, 3]})
#print(tgdf)

#print(deldf)
D = pd.DataFrame({'OptPrem':greeks[:, 0], 'Delta':greeks[:, 1], 'Gamma':greeks[:, 2],
                  'Theta':greeks[:, 3], 'Vega':greeks[:, 5], 'PNL Option':greeks[:, 6],
                  'StrikePrice':greeks[:, 7]})
#print(D)
output = pd.concat([df,D], axis = 1)
#print(output)
output.to_excel(r'/Users/piersonmichalak/Desktop/test.xlsx',sheet_name ='datastuff',
             startrow=6, index = True)