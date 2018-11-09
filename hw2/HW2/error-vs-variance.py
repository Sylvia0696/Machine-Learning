import numpy as np
import math
from scipy.stats import pearsonr
import statsmodels.api as sm   
import random
import matplotlib.pyplot as plt


def empirical_risk(data,labels,n,e_y):
    regr = sm.OLS(labels,data) ##OLS
    res = regr.fit()   ##outcome
    res.params.resize(res.params.shape[0],1)
    risk = (np.dot(data,res.params) - e_y)**2
    risk = np.true_divide(risk,n)
    risk = np.sum(risk)
    return risk


def k_times_empirical_risk(k,n):
    z = np.linspace(-math.pi,math.pi,n)
    z.resize(n,1)
    e = np.random.randn(n,1)
    y = np.sin(3*z/2) + e
    e_y = np.sin(3*z/2)
    x = np.ones((n,1))
    col = []
    for i in range(1,k+1):
        tmp_m = z**i
        x = np.c_[x,tmp_m]
        risk = empirical_risk(x,y,n,e_y)
        #print("k=",i,"empirical risk=",risk)
        col.append(risk)
    return col



if __name__ == '__main__':
    k = 20
    n = 1001
    k_risk = []
    for i in range(1,1001):
        tmp_c = k_times_empirical_risk(k,n)
        k_risk.append(tmp_c)
    k_risk = np.array(k_risk)
    k_risk = np.sum(k_risk,axis=0)
    k_risk = k_risk/1000
    print(k_risk)

    k = list(range(1,21))
    plt.scatter(k,k_risk)
    plt.xlabel("k")
    plt.ylabel("risk")
    plt.plot(k,k_risk)