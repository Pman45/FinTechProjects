import pandas as pd
import numpy as np
import scipy.stats as sci
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import openpyxl
import datetime
from sklearn import linear_model
import io
from tiingo import TiingoClient

config = {}
config['session'] = True
config['api_key'] = "92d790d9ad4704559ef11ce7c8fe9f9810b355ea"
client = TiingoClient(config)

UTILtix = ['NEE','AWK','ES','XLU','ENIA','SPKE','SPH','PPL','NGG','TRP',
           'SJI','BIP','AQN','DUK','AVA','NEW']

MLPtix = ['AMLP','MLPA', 'MLPX', 'MLPB', 'ATMP', 'AMZA','AMUB','MLPI'
    ,'MLPY','BMLP','MLPC','ZMLP','IMLP']

AEROtix = ['ITA', 'XAR', 'DFEN', 'FITE','LMT', 'XLI', 'TDG', 'TDY', 'GD','RTX']

BFINtix = ['XLF','VFH','EUFN','FNCL','FXO','UYG','IXG','RYF',
           'GREK','DFNL','SKF','PFI','JHMF']
BigList = [UTILtix, MLPtix, AEROtix, BFINtix]
def findpairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.02:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs
# namelist = []
# for i in BigList:
#     pairs = findpairs(i)
#     namelist.append('i' + pairs)
#     print(pairs)

def closedata():
    px = client.get_dataframe(MLPtix,
                                frequency='daily',
                                metric_name='adjClose',
                                startDate='2018-01-01',
                                endDate='2020-07-29')
    px.columns = MLPtix
    #corr = px.corr().unstac().sort_values().drop_duplicates()
    #print(corr)
    checknan = px.isnull().values.any()
    px2 = px.dropna(axis='columns')
    return px, checknan, px2

print(closedata())