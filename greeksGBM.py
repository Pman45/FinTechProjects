import pandas as pd
import scipy.stats as stats
import numpy as np
import math
import multiprocessing as mp
import datetime
from datetime import timedelta
import sys
np.set_printoptions(threshold=sys.maxsize)

S0 = 100#float(input("S0: "))
K =  100#float(input("K: "))
T =  0.25#float(input("T, time in years/period end: ")) #time in years,
r = 0.0001 # float(input("r, constant short rate: ")) #constant short rate
sigma = float(0.2)#float(input("sigma: ")) #volatility
M = 63#int(input("M, number of time steps:")) #number of steps within each simulation
I = 10000#int(input("I, number of paths:")) #numbr of paths/simulations


S = np.zeros([I,M])
t = range(0,M,1)
    #the number of time steps and paths
n1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T)/(sigma * np.sqrt(T))
n2 = (np.log(S0/K) + (r-0.5*sigma**2)*T)/(sigma * np.sqrt(T))
q = 0  # q is the dividend risk rate
delta = np.exp(-q * T) * stats.norm.cdf(n1, 0.0, 1.0)
gamma = stats.norm.cdf(n1,0.0,1.0)/(S0*T**0.5)
theta = (-S0*stats.norm.pdf(n1,0.0,1.0)*sigma)/(2*np.sqrt(T))-r*K*np.exp(-r*T)*stats.norm.cdf(-n1,0.0,1.0)
vega = S0*stats.norm.pdf(n1,0.0,1.0)*np.sqrt(T)
rho = K*T*np.exp(-r*T)*stats.norm.cdf(n2,0.0,1.0)

def gen_paths(S0, r, sigma, T, M, I):
    dt = float(T) / M
    paths = np.zeros((M + 1, I), np.float64)
    paths[0] = S0
    for t in range(1, M + 1):
        rand = np.random.standard_normal(I)
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt +
                                         sigma * np.sqrt(dt) * rand)

    return paths

def rvoldata(S0,r,sigma, T,M,I):
    dt = float(T)/M
    ret = np.zeros((M+1,I))
    ret[0][0] = S0
    ret[0][1] = 0
    for t in range(1,M+1):
        rand = np.random.standard_normal(I)
        ret[t][0] = ret[t-1][0]*np.exp((r-0.5*sigma**2)*dt + sigma*np.sqrt(dt))
        ret[t][1] = np.log(ret[t][0]/ret[t-1][0])

    Rvol = 100*np.sqrt(252/(M+1)*np.sum((ret[:, 1])**2))

    return Rvol
def greeksdata(S0,r,sigma,T,M,I,K):
    dt = float(T)/M

    greeks = np.zeros((M+1,7), np.float64 )
    greeks[0][0] = S0
    greeks[0][1]= delta#eurocalloption._init_.
    greeks[0][2]= gamma#eurocalloption._init_.self.
    greeks[0][3]= theta#eurocalloption._init_.self.
    greeks[0][4]= rho#eurocalloption._init_.self.
    greeks[0][5]= vega#eurocalloption._init_.self.
    greeks[0][6] = 0

    #(np.log(greeks[0][0] / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau)) = n1
    #(np.log(greeks[0][0] / K) - (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau)) = n2
    for i in range(1,M+1):
        tau = T - i * dt
        greeks[i][0] = greeks[i - 1][0] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(tau) )
        d1 = (np.log(greeks[i][0] / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
        d2 = (np.log(greeks[i][0] / K) - (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))

        greeks[i][1] = stats.norm.cdf(d1,0.0,1.0)
        greeks[i][2] = stats.norm.cdf(d1,0.0,1.0)/(greeks[i][0]*T**0.5)
        greeks[i][3] = (-greeks[i][0]*stats.norm.pdf(d1,0.0,1.0)*sigma)/(2*np.sqrt(tau))\
                       -r*K*np.exp(-r*tau)*stats.norm.cdf(-d1,0.0,1.0)
        greeks[i][4] = K*T*np.exp(-r*tau)*stats.norm.cdf(d2,0.0,1.0)
        greeks[i][5] = greeks[i][0]*stats.norm.pdf(d1,0.0,1.0)*np.sqrt(tau)
        greeks[i][6] = ((greeks[i][0]*stats.norm.cdf(d1,0.0,1.0) - K*np.exp(-r*tau)*stats.norm.cdf(d2,0.0,1.0)
                         - greeks[i-1][0]*stats.norm.cdf((np.log(greeks[i-1][0] / K)
                        + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau)),0.0,1.0) +K*np.exp(-r*tau)*stats.norm
                         .cdf((np.log(greeks[i-1][0] / K) - (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau)),0.0
                              ,1.0))) - greeks[i][1]*(greeks[i][0]-greeks[i-1][0])
    return greeks



np.random.seed(123)
paths = gen_paths(S0, r, sigma, T, M, I)
paths[:, 0].round(4)
#print paths
np.average(paths[-1])

np.random.seed(123)
greeks = greeksdata(S0,r,sigma,T,M,I,K)
greeks[:, 0].round(4)


#df=pd.DataFrame( np.array([greeks[:, 0],greeks[:, 1],greeks[:, 2],greeks[:, 3], greeks[:, 4], greeks[:, 5]]),
#                index = range(64),columns=['Price', 'Delta', 'Gamma', 'Theta','Rho', 'Vega'])
#df=pd.DataFrame( np.array([greeks[:,0],greeks[:, 1],greeks[:, 2],greeks[:, 3], greeks[:, 4], greeks[:, 5]]),
                #index = range(64),columns=['Price', 'Delta', 'Gamma', 'Theta','Rho', 'Vega'])
d={'Price':[greeks[:, 0]], 'Delta':[greeks[:, 1]], 'Gamma':[greeks[:, 2]], 'Theta':[greeks[:, 3]]
, 'Rho':[greeks[:, 4]], 'Vega':[greeks[:, 5]], 'Pnl':[greeks[:, 6]]}
df = pd.DataFrame(data=d)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df)
#print('Delta', np.average(greeks[-1][1],',Gamma',np.average(greeks[-1][2])),',Theta',np.average(greeks[-1][3]),'Rho',
      #np.average(greeks[-1][4]),',Vega',np.average(greeks[-1][5]))

CallPayoffAverage = np.average(np.maximum(0, paths[-1] - K))
CallPayoff = np.exp(-r*T) * CallPayoffAverage
print('Your finall call payoff with GBM is',CallPayoff)

print('The realized volatility is', rvoldata(S0,r,sigma, T,M,I))




#class eurocalloption:

 #   def d1(self, S0, K,r, dt,sigma):
  #      return (math.log((S0/K))+(r+math.pow(sigma,2)/2)*dt)/(sigma*math.sqrt(dt))
   # def d2(self,d1,sigma,dt):
    #    return d1-(sigma*math.sqrt(dt))
    #def path_price(self,S0,K,r,sigma,dt,d1,d2):
    #    n1 = stats.norm.cdf(d1)
    #    n2 = stats.norm.cdf(d2)
    #    return S0*n1-K*(np.exp(-r*dt))*n2
    #def delta(self,d1):
    #    return stats.norm.cdf(d1)
    #def vega(self,S0,d1,d2,dt):
    #    return S0*stats.norm.cdf(d1,0.0,1.0)*np.sqrt(dt)
    #def gamma(self,S0,d1,dt):
    #    return stats.norm.cdf(d1,0.0,1.0)/(S0*T**0.5)
    #def theta(self,S0,d1,dt,sigma,K,r):
    #    return-(S0*stats.norm.pdf(d1,0.0,1.0)*sigma)/(2*np.sqrt(dt))-r*K*np.exp(-r*dt)*stats.norm.cdf(-d1,0.0,1.0)
    #def rho(self,K,dt,r,d2):
    #    return K*dt*np.exp(-r*dt)*stats.norm.cdf(d2,0.0,1.0)

    #def _init_(self, S0, K, sigma, T, r):
     #   self.S0 = S0
     #   self.K = K
     #   self.sigma = sigma
     #   self.T = T
     #   self.r = r
     #   range=range(1,T)
     #   for i in range:
     #       dt=float(T-i)/M

      #  d1 = self.d1(S0, K, r, sigma, dt)

       # d2 = self.d2(d1, sigma, dt)
       # self.dt = dt
       # self.price = self.price(S0, d1, K, d2, r, dt)
       # self.delta = self.delta(d1)
       # self.gamma = self.gamma(self,S0,d1,dt)
       # self.vega = self.vega(self,S0,d1,d2,dt)
       # self.theta = self.theta(self,S0,d1,dt,sigma,K,r)
       # self.rho = self.rho(self,K,dt,r,d2)

#class price_data:
    #def time_step(self, M,T):
       # for i in range(1,T)
          #  dt = (T-i)/M
         #   if dt != 0:
         #       eo = EuropeanCall(self.S0[self.index] + np.random.normal(0, dt ** (1 / 2)), self.K,
         #                         self.sigma, self.T, self.r)
          #  self.option_prices.append(eo.price)
          #  self.delta.append(eo.delta)
         #   self.gamma.append(eo.gamma)
         #   self.theta.append(eo.theta)
         #   self.rho.append(eo.rho)
         #   self.vega.append(eo.vega)
         #   self.index_set.append(self.index)