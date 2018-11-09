from __future__ import print_function
from scipy.io import loadmat
import numpy as np

def estimate_naive_bayes_classifier(X,Y,yn):
    '''divide X into yn classes'''
    list_x = []
    for i in range(1,yn+1):
        idx = [idx for idx, elem in enumerate(Y) if elem == i]
        list_x.append(X[idx,:])

    '''count mu and pi'''
    n,d = X.shape
    mu = np.zeros((yn,d))
    pi = np.zeros((yn,1))

    for i in range(0,yn):
        darray = list_x[i]
        sumx = np.sum(darray, axis = 0)
        piy = darray.shape[0]
        mu[i,:] = np.divide((1+sumx),(2+piy))
        pi[i,0] = np.divide(float(piy),float(yn))

    return (mu,pi)


def predict(params,X,yn):
    mu = params[0]
    pi = params[1]

    (n,d) = X.shape
    pred_matrix = np.zeros((n,yn))
   
    #count 1-X, as X is sparse matrix
    data_X = np.tile([1],(n,d)) - X
    pred_matrix = np.matmul(np.log(mu),X.T) + np.matmul(np.log(1 - mu),data_X.T)
    pred_mt = np.zeros((yn,n))
    for i in range(yn):
        pred_mt[i,:] = np.log(pi[i,0]) + pred_matrix[i,:]

    pred = np.zeros((n,1))
    pred[:,0] = np.argmax(pred_mt,axis=0)
    pred = pred + 1
    
    return pred


def print_top_words(params,vocab):
    
    mu = params[0]
    x0 = mu[1,:] * (1 - mu[0,:])
    x1 = mu[0,:] * (1 - mu[1,:])
    alpha = np.log(x0 / x1)
    
    d = len(vocab)
    alpha_idx = sorted(range(d), key=lambda k: alpha[k])
    minnum = alpha_idx[0:20]
    maxnum = alpha_idx[-20:]
    min_list = []
    max_list = []
    for i in minnum:
        min_list.insert(0,vocab[i])
    for i in maxnum:
        max_list.insert(0,vocab[i])
    print("largest words are",max_list)
    print("smallest words are",min_list)




def load_data():
    return loadmat('news.mat')




def load_vocab():
    with open('news.vocab') as f:
        vocab = [ x.strip() for x in f.readlines() ]
    return vocab



if __name__ == '__main__':
    news = load_data()
    # 20-way classification problem

    data = news['data']
    labels = news['labels']
    testdata = news['testdata']
    testlabels = news['testlabels']
    
    data = data.toarray()
    testdata = testdata.toarray()
#     print(data.shape)
#     print(labels.shape)
#     print(params.shape)
    params = estimate_naive_bayes_classifier(data,labels,20)
    pred = predict(params,data,20) # predictions on training data
    testpred = predict(params,testdata,20) # predictions on test data

    print('20 classes: training error rate: %g' % np.mean(pred != labels))
    print('20 classes: test error rate: %g' % np.mean(testpred != testlabels))
    
    (ntrain,dtrain) = data.shape
    (ntest,dtest) = testdata.shape
    
    # 2 classes
    indices = (labels==1) | (labels==16) | (labels==20) | (labels==17) | (labels==18) | (labels==19)  
    indices = np.reshape(indices, ntrain)
    data2 = data[indices,:]
    labels2 = labels[indices]
    labels2[(labels2==1) | (labels2==16) | (labels2==20)] = 1
    labels2[(labels2==17) | (labels2==18) | (labels2==19)] = 2
    
    testindices = (testlabels==1) | (testlabels==16) | (testlabels==20) | (testlabels==17) | (testlabels==18) | (testlabels==19)
    testindices = np.reshape(testindices, ntest)
    testdata2 = testdata[testindices,:]
    testlabels2 = testlabels[testindices]
    testlabels2[(testlabels2==1) | (testlabels2==16) | (testlabels2==20)] = 1
    testlabels2[(testlabels2==17) | (testlabels2==18) | (testlabels2==19)] = 2

    params2 = estimate_naive_bayes_classifier(data2,labels2,2)
    pred2 = predict(params2,data2,2) # predictions on training data
    testpred2 = predict(params2,testdata2,2) # predictions on test data
    print('2 classes: training error rate: %g' % np.mean(pred2 != labels2))
    print('2 classes: test error rate: %g' % np.mean(testpred2 != testlabels2))

    vocab = load_vocab()
    print_top_words(params2,vocab)