import pandas as pd
import numpy as np
import sys
import math
import multiprocessing as mp
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as si
numpy.set_printoptions(threshold=sys.maxsize)

S0 = float(input("S0: "))
K = float(input("K: "))
T = float(input("T, time in years/period end: ")) #time in years,
r = float(input("r, constant short rate: ")) #constant short rate
sigma = float(input("sigma: ")) #volatility
M = int(input("M, number of time steps:")) #number of steps within each simulation
I = int(input("I, number of paths:")) #numbr of paths/simulations


S = np.zeros([I,M])
t = range(0,M,1)

    #the number of time steps and paths
def gen_paths(S0, r, sigma, T, M, I):
    dt = float(T) / M
    paths = np.zeros((M + 1, I), np.float64)
    paths[0] = S0
    for t in range(1, M + 1):
        rand = np.random.standard_normal(I)
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt +
                                         sigma * np.sqrt(dt) * rand)
    return paths

np.random.seed(123)
paths = gen_paths(S0, r, sigma, T, M, I)
log_returns = np.log(paths[1:] / paths[0:-1])

paths[:, 0].round(4)
np.average(paths[-1])

print paths

CallPayoffAverage = np.average(np.maximum(0, paths[-1] - K))
CallPayoff = np.exp(-r*T) * CallPayoffAverage
print('Your finall call payoff with GBM is',CallPayoff)

def bsm_price(S0,T,r,K,sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    price = (S0 * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    return price
print('Via BSM, the price is ',bsm_price(S0,T,r,K,sigma))

print('Percent difference:',(abs((bsm_price(S0,T,r,K,sigma))-CallPayoff))/100 ,'%')