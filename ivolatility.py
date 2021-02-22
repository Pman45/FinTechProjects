import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve
import numpy as np
from math import sqrt, exp, log, pi
import scipy.stats as si
from scipy.stats import norm
# S0= underlying price
# K = strike price
# r= riskless short rate
# T= time to maturity (as percentage of a year)
# sigma = implied volatility
# C0 = initial options call price
# P0= initial options put price

#we define the following from Black-Scholes
S0 = float(input("S0: "))
K = float(input("K: "))
T = (float(input("T : ")))
r = float(input("r: "))
C0 = float(input("C0: "))
#P0 = float(input("P0: "))
sigma = float(input("sigma: "))



def d(sigma, S0, K, r, T):
    d1 = (log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = (log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    return d1,d2

def call_price(sigma, S0, K, r, T, d1, d2):
    C = S0*si.norm.cdf(d1,0.0,1.0) - K*si.norm.cdf(d2,0.0,1.0)*np.exp(-r*T)
    return C

def put_price(sigma, S0, K, r, T, d1, d2):
    P = -si.norm.cdf(d1)*S0 + si.norm.cdf(d2)*K*np.exp(-r*T)
    return P

def f(sigma,S0,K,r,T,d1,d2,C0): #f(x) =C(x) -P0, where C(x) is the call price given by BSM in terms
    # of the volatility. Using Newton's approximation we can determine the implied volatility
    valuef = call_price(sigma,S0,K,r,T,d1,d2) - C0
    return valuef

def volcall(S0,K,T,r,C0,sigma):
    #call_price(sigma, S0, K, r, T, d1, d2)
    #tolerances and variations of volalitily between iterations
    d1, d2 = d(sigma, S0, K, r, T)
    valuef =  call_price(sigma,S0,K,r,T,d1,d2) - C0
    V = S0 * norm.pdf(d1) * sqrt(T)

    tolerance = 0.000001
    vol0 = sigma
    newvol = vol0
    oldvol= vol0 - 1
    epsilon = newvol-oldvol

    while abs(epsilon) > tolerance:
        #if count >= max:
        #    print('Error. Count broken')
        #    break;
        oldvol = newvol
        newvol = (newvol - valuef - C0)/V
        return abs(newvol)

print('Implied volatility is',volcall(S0,K,T,r,C0,sigma))






