import numpy as np   ##科学计算库 
import pandas as pd
import matplotlib.pyplot as plt  ##绘图库
import statsmodels.api as sm   ##引入最小二乘法算法
from scipy.io import loadmat
import math
from scipy.stats import pearsonr





def ordinary_least_square(label,data):
    regr = sm.OLS(label,data) ##OLS
    res = regr.fit()   ##outcome
    return res.params


def squared_loss(testdata,testlabel,params):
    y = np.dot(testdata,params)  ##test_data_label
    y.resize(y.shape[0],1)
    square = (y-testlabel)*(y-testlabel)  
    squared_loss = np.sum(square,axis = 0)   ##squared_loss of testdata
#     print(squared_loss)
    n = testdata.shape[0]
    return squared_loss/n


def sparse_linear_predictor(data,label,testdata,testlabel):
    lmin = math.inf
    name = []
    min_params = []
    for i in range(1,data.shape[1]):
        for j in range(i+1,data.shape[1]):
            for k in range(j+1,data.shape[1]):
                new_data = data[:,[0,i,j,k]]
                params = ordinary_least_square(label,new_data)
                squaredloss = squared_loss(new_data,label,params)
#                 print(squaredloss)
                if lmin > squaredloss:
                    lmin = min(lmin,squaredloss)
                    name = [0,i,j,k]
#                     print(name)
                    min_params = params
    new_testdata = testdata[:,name]
    test_squareloss = squared_loss(new_testdata,testlabel,min_params)
    return name, min_params, test_squareloss


def pearson_correlation(testdata,name):
    name_value = []
    for i in range(1,len(name)):
        count = 1
        nv_list = []
        for j in range(1,testdata.shape[1]):
            cor = pearsonr(testdata[:,name[i]],testdata[:,j])
            if name[i] == j:
                continue
            elif count == 1:
                nv_list.append(j)
                nv_list.append(cor[0])
                count += 1
            elif count == 2:
                if abs(cor[0]) > abs(nv_list[1]):
                    nv_list.insert(0,j)
                    nv_list.insert(1,cor[0])
                else:
                    nv_list.append(j)
                    nv_list.append(cor[0])
                count += 1
            elif count > 2:
                if abs(cor[0]) > abs(nv_list[1]):
                    nv_list.insert(0,j)
                    nv_list.insert(1,cor[0])
                    del nv_list[4:]
                elif abs(cor[0]) < abs(nv_list[1]) and abs(cor[0]) > abs(nv_list[3]):
                    nv_list[2] = j
                    nv_list[3] = cor[0]

        name_value.append(nv_list)
    return name_value



if __name__ == '__main__':
    wine = loadmat('wine.mat')
    data = wine['data']
    label = wine['labels']
    testdata = wine['testdata']
    testlabel = wine['testlabels']

    params = ordinary_least_square(label,data)
    squareloss = squared_loss(testdata,testlabel,params)
    print("Test risks of the ordinary least squares estimator", squareloss)

    res = sparse_linear_predictor(data,label,testdata,testlabel)
    print("Test risks of the sparse linear predictor", res[2])
    print("Names_ID and value",res[0],res[1])

    name_value = pearson_correlation(testdata,res[0])
    print("nameID_value",name_value)