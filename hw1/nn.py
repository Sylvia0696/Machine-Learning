#!/usr/bin/env python

from __future__ import print_function
from scipy.io import loadmat
from random import sample
import numpy as np
import matplotlib.pyplot as plt

def nn(X,Y,test):

    x2 = np.square(X).sum(axis = 1)                                     
    xt = np.dot(test,X.T)
    t2 = np.transpose(np.square(test).sum(axis = 1)[np.newaxis]) 
    dist = x2 - 2*xt + t2
    preds = Y[np.argmin(dist, axis = 1)]
    return preds

if __name__ == '__main__':
    ocr = loadmat('ocr.mat')
    num_trials = 10
    mean_error = []
    std_err = []
    num = [1000,2000,4000,8000]
    for n in [ 1000, 2000, 4000, 8000 ]:
        test_err = np.zeros(num_trials)
        for trial in range(num_trials):
            sel = sample(range(len(ocr['data'].astype('float'))),n)
            preds = nn(ocr['data'].astype('float')[sel], ocr['labels'][sel], ocr['testdata'].astype('float'))
            test_err[trial] = np.mean(preds != ocr['testlabels'])
        print("%d\t%g\t%g" % (n,np.mean(test_err),np.std(test_err)))
        mean_error.append(np.mean(test_err))
        std_err.append(np.std(test_err))
        
    plt.scatter(num,mean_error)
    plt.errorbar(num,mean_error,std_err)
    plt.title('Learning Curve Plot')
    plt.xlabel('n')
    plt.ylabel('Error rate')
    plt.show()