from _ast import In
from math import log, sqrt, exp
from scipy import stats

#for subscripts you can use either "xx".translate(SUB) or \u208n where n is the number
S0 = float(input("S0: "))
K = float(input("K: "))
T = float(input("T: "))
r = float(input("r: "))
sigma = float(input("sigma: "))

def bsmc_value(S0,T,r,sigma,K ):
    """ Valuation of European call option in BSM model.
     Analytical formula.
     Parameters
     ==========
     S0 : float
     initial stock/index level
     K : float
     strike price
     T : float
     maturity date (in year fractions)
     r : float
     constant risk-free short rate
     sigma : float
     volatility factor in diffusion term
     Returns
     =======
     value : float
     present value of the European call option
     """

    d1 = ((log(S0/K)) + ((r + 0.5 * sigma ** 2)*T))/(sigma * sqrt(T))
    d2 = ((log(S0/K)) + ((r - 0.5*sigma **2)*T))/(sigma * sqrt(T))
    cprice=(S0*stats.norm.cdf(d1,0.0,1.0))-K*(exp(-r*T)*stats.norm.cdf(d2,0.0,1.0))
    return d1, cprice

#def bsm_vega(S0,K,r,T,sigma): #vega corresponds to the partial derivative of BSM w.r.t sigma

   # S0 = float(S0)
   # d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T / (sigma * sqrt(T))
   # V = S0 * stats.norm.cdf(d1, 0.0, 1.0) * sqrt(T)
   # return V

def bsmc_impvol(S0,K,T,r,C0,sigma_approx, it=100):
    #i is the integer number of iterations
    C0 = cprice
    for i in range(it):
        sigma_est -= ((bsm_call_value(S0, K, T, r, sigma_est) - C0)
                      / bsm_vega(S0, K, T, r, sigma_est))
        return sigma_est

print ('The call option has price', bsmc_value(S0,T,r,K,sigma))
#print ('Vega has value',vega )
#print('Implied volatility is', sigma_est)