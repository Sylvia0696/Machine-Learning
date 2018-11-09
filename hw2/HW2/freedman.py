import numpy as np   ##科学计算库 
from scipy.io import loadmat
import math
from scipy.stats import pearsonr
import statsmodels.api as sm   
import random


def empirical_risk(data,labels):
    regr = sm.OLS(labels,data) ##OLS
    res = regr.fit()   ##outcome
    n = data.shape[0]
    res.params.resize(res.params.shape[0],1)
    risk_jhat = (np.dot(data,res.params)-labels)**2
    risk_jhat = np.true_divide(risk_jhat,n)
    risk_jhat = np.sum(risk_jhat)
    return risk_jhat




def freedman_risk(data,labels):
    n = data.shape[0]
    pj_hat = np.true_divide(np.dot(data.T,labels),n)
    j_hat = []
    cols_j = []
    for i in range(0,len(pj_hat)):
        if abs(pj_hat[i]) > 1.75/math.sqrt(n):
            j_hat.append(pj_hat[i].tolist())
            cols_j.append(i)
    j_hat = np.array(j_hat)

    new_data = data[:,cols_j]
    f_risk = empirical_risk(new_data,labels)
    return f_risk,len(cols_j)



def random_data():
    n = random.randint(200,1000)
    d = random.randint(200,1000)
    data_r = np.random.randn(n,d)
    label_r = np.random.randn(n,1)    
    risk = freedman_risk(data_r,label_r)
    data_rnew = data_r[:,0:risk[1]]
    risk_e = empirical_risk(data_rnew,label_r)    
    print("n is",n,"; d is",d,"; random risk is",risk[0],"; Irisk is",risk_e)





if __name__ == '__main__':
    freedman = loadmat('freedman.mat')
    data = freedman['data']
    labels = freedman['labels']

    len_cols = freedman_risk(data,labels)
    print("features remain after the screening is",len_cols[1])
    print("the empirical risk of β(J) is",len_cols[0])

    data_c = data[:,0:75]
    risk_c = empirical_risk(data_c,labels)
    print("the empirical risk of β(I) is",risk_c)

    random_data()
    random_data()
    random_data()
    random_data()
    random_data()
    random_data()
    random_data()
    random_data()
    random_data()
    random_data()