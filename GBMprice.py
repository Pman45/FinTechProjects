import pandas as pd
import numpy as np
import math
import multiprocessing as mp
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as scs

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

np.average(paths[-1])

CallPayoffAverage = np.average(np.maximum(0, paths[-1] - K))
CallPayoff = np.exp(-r*T) * CallPayoffAverage
print('Your finall call payoff is',CallPayoff)

def bsm_price(S0,T,r,K,sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    price = (S0 * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    return price
print price


ST1 = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * np.random.standard_normal(I))
ST2 = S0 * np.random.lognormal((r - 0.5 * sigma ** 2) * T,sigma * np.sqrt(T), size=I)

def print_stats(ST1,ST2):
    #hopefully displays arrays for ST1,2

     sta1 = scs.describe(ST1)
     sta2 = scs.describe(ST2)
     print ("%14s %14s %14s" % \
     ('statistic', 'Normal Distribution', 'Lognormal Distribution'))
     print (45 * "-")
     print ("%14s %14.3f %14.3f" % ('size', sta1[0], sta2[0]))
     print ("%14s %14.3f %14.3f" % ('min price', sta1[1][0], sta2[1][0]))
     print ("%14s %14.3f %14.3f" % ('max price ', sta1[1][1], sta2[1][1]))
     print ("%14s %14.3f %14.3f" % ('mean price', sta1[2], sta2[2]))
     print ("%14s %14.3f %14.3f" % ('std', np.sqrt(sta1[3]),
     np.sqrt(sta2[3])))

print(print_stats(ST1,ST2))

plt.hist(ST1, bins=50)
plt.xlabel('index level')
plt.ylabel('frequency')
plt.grid(True)

plt.hist(ST2, bins=50)
plt.xlabel('index level')
plt.ylabel('frequency')
plt.grid(True)

plt.show()

