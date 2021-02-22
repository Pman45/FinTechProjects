import numpy as np
import scipy.stats as si
import sympy as sy
from sympy.stats import Normal, cdf
from sympy import init_printing
init_printing()
#Parameters
# ==========
# S0 : float
# initial stock/index level
# K : float
# strike price
# T : float
# time to maturity
# r : float
# constant risk-free short rate
# sigma : float
# volatility factor in diffusion term
# Returns
# =======

S0 = float(input("S0: "))
K = float(input("K: "))
T = float(input("T: "))
r = float(input("r: "))
sigma = float(input("sigma: "))
option = input("call or put option:")
def bsmeurovanilla(S0,K,r,T,sigma,option):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T)/(sigma * np.sqrt(T))
    d2 = (np.log(S0/K) + (r-0.5*sigma**2)*T)/(sigma * np.sqrt(T))

    if option == 'call':
        price = (S0*si.norm.cdf(d1,0.0,1.0) - K*np.exp(-r*T)*si.norm.cdf(d2,0.0,1.0))
    if option == 'put':
        price = (-S0*si.norm.cdf(-d1,0.0,1.0) + K*np.exp(-r*T)*si.norm.cdf(-d2,0.0,1.0))
    return price

print('The option price is',bsmeurovanilla(S0,K,r,T,sigma,option), 'dollars')

def deltacall(S0,K,r,T,sigma,option):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    q = 0 #q is the dividend risk rate
    delta = np.exp(-q*T)*si.norm.cdf(d1,0.0,1.0)
    return delta
def gammacall(S0,K,r,T,sigma,option):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = si.norm.cdf(d1,0.0,1.0)/(S0*T**0.5)
    return gamma
def thetacall(S0, K, r, T, sigma, option):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    theta = -(S0*si.norm.pdf(d1,0.0,1.0)*sigma)/(2*np.sqrt(T))-r*K*np.exp(-r*T)*si.norm.cdf(-d1,0.0,1.0)
    return theta
def vegacall(S0,K,r,T,sigma,option):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    vega = S0*si.norm.pdf(d1,0.0,1.0)*np.sqrt(T)
    return vega
def rhocall(S0,K,r,T,sigma,option):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    rho = K*T*np.exp(-r*T)*si.norm.cdf(d2,0.0,1.0)
    return rho

print('The induced greeks for the call option are','delta',deltacall(S0,K,r,T,sigma,option),'gamma',gammacall(S0,K,r,T,sigma,option),
      'theta',thetacall(S0,K,r,T,sigma,option),'vega',vegacall(S0,K,r,T,sigma,option),'rho',rhocall(S0,K,r,T,sigma,option))
