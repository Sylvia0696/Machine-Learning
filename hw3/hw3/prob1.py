import csv
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from toolz import itertoolz, compose
from toolz.curried import map as cmap, sliding_window, pluck


# functions

def first_round(data,labels):
    w = np.zeros((data.shape[1]+1,1))
    for i in range(data.shape[0]):
        tmp = (data[i] * w[1:] + w[0]) * labels[i][0]
        if tmp <= 0:
            w[1:] += labels[i][0] * data[i].T
            w[0] += labels[i]
    return w


def second_round(data,labels,w):
    countlist = list(range(data.shape[0]))
    random.shuffle(countlist)
    w_final = w / (data.shape[0]+1)
    for i in countlist:
        tmp = (data[i] * w[1:] + w[0]) * labels[i][0]
        if tmp <= 0:
            w[1:] += labels[i][0] * data[i].T
            w[0] += labels[i]
        w_final += w / (data.shape[0]+1)
    return w_final


def count_error(data,labels,w_final):
    count = 0
    for i in range(data.shape[0]):
        tmp = (data[i] * w_final[1:] + w_final[0]) * labels[i][0]
        if tmp <= 0:
            count += 1
    error = float(count / data.shape[0])
    return error


# import data

train = pd.read_csv('reviews_tr.csv')
text = train['text']
labels = train['rating']
vectorizer = CountVectorizer()
data = vectorizer.fit_transform(text)
labels = np.array(labels)
labels = labels.reshape(labels.shape[0],1)
for i in range(labels.shape[0]):
    if labels[i] == 0:
        labels[i] = -1



test = pd.read_csv('reviews_te.csv')
test_text = test['text']
testlabels = test['rating']
testdata = vectorizer.transform(test_text)
testlabels = np.array(testlabels)
testlabels = testlabels.reshape(testlabels.shape[0],1)
for i in range(testlabels.shape[0]):
    if testlabels[i] == 0:
        testlabels[i] = -1




# unigram error rate

w = first_round(data,labels)
w_final = second_round(data,labels,w)
error = count_error(data,labels,w_final)
print(error)
error_test = count_error(testdata,testlabels,w_final)
print(error_test)



# high and low words

w_list = []
for i in range(w_final.shape[0]):
    w_list.append(w_final[i][0])
word = vectorizer.get_feature_names()
idxh = np.argpartition(w_list, -10)[-10:]
idxl = np.argpartition(w_list, 10)[:10]
high = []
for i in idxh:
    high.append(word[i])
low = []
for i in idxl:
    low.append(word[i])
high.sort()
low.sort()
print(high)
print(low)





# high and low words in 2 texts

count = 0
idx = []
countlist = list(range(data.shape[0]))
random.shuffle(countlist)

for i in countlist: 
    if count < 2:
        tmp = (data[i] * w[1:] + w[0]) * labels[i][0]
        if tmp <= 0:
            count += 1
            print(text[i])
            datatmp = data[i].toarray()
            #print(type(datatmp))
            w_list2 = []
            for i in range(w.shape[0]):
                w_list2.append(w[i][0])
            wx = np.multiply(datatmp, w_list2[1:])
            #print(type(wx))
            #print(wx)
            wx = wx.tolist()
            wx = wx[0]
            print(type(wx))
            idxh = np.argpartition(wx, -10)[-10:]
            idxl = np.argpartition(wx, 10)[:10]
            print(type(idxh))
            #print(idxh)
            high2 = []
            highvalue = []
            for i in idxh:
                high2.append(word[i])
                highvalue.append(wx[i])
            low2 = []
            lowvalue = []
            for i in idxl:
                low2.append(word[i])
                lowvalue.append(wx[i])
            print(high2)
            print(highvalue)
            print(low2)
            print(lowvalue)
            w[1:] += labels[i][0] * data[i].T
            w[0] += labels[i]
    else:
        break




# idf error rate

vectorizer_idf = TfidfVectorizer()
data_idf = vectorizer_idf.fit_transform(text)
w_idf = first_round(data_idf,labels)
w_final_idf = second_round(data_idf,labels,w_idf)
error_idf_tr = count_error(data_idf,labels,w_final_idf)
print(error_idf_tr)
testdata_idf = vectorizer_idf.transform(test_text)
error_idf_te = count_error(testdata_idf,testlabels,w_final_idf)
print(error_idf_te)






# bigram error rate

bigram = CountVectorizer(ngram_range=(2, 2))
data_bi = bigram.fit_transform(text)
w_bi = first_round(data_bi,labels)
w_final_bi = second_round(data_bi,labels,w_bi)
error_bi_tr = count_error(data_bi,labels,w_final_bi)
print(error_bi_tr)
testdata_bi = bigram.transform(test_text)
error_bi_te = count_error(testdata_bi,testlabels,w_final_bi)
print(error_bi_te)




# skip-grams error rate

class SkipGramVectorizer(CountVectorizer):
    def build_analyzer(self):    
        preprocess = self.build_preprocessor()
        stop_words = self.get_stop_words()
        tokenize = self.build_tokenizer()
        return lambda doc: self._word_skip_grams(
                compose(tokenize, preprocess, self.decode)(doc),
                stop_words)

    def _word_skip_grams(self, tokens, stop_words=None):
        # handle stop words
        if stop_words is not None:
            tokens = [w for w in tokens if w not in stop_words]

        return compose(cmap(' '.join), pluck([0, 2]), sliding_window(3))(tokens)

vect = SkipGramVectorizer()
data_skip = vect.fit_transform(text)
w_skip = first_round(data_skip,labels)
w_final_skip = second_round(data_skip,labels,w_skip)
error_skip_tr = count_error(data_skip,labels,w_final_skip)
print(error_skip_tr)
testdata_skip = vect.transform(test_text)
error_skip_te = count_error(testdata_skip,testlabels,w_final_skip)
print(error_skip_te)