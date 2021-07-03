import nltk.data
import codecs
import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import euclidean_distances
from gensim.models import Word2Vec
from pyemd import emd
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import _stop_words
import random
import numpy as np
import random as rn
import numpy as np
import matplotlib.pyplot as plt
from random import randrange, uniform
#from nsgacrowdingdis import *
import numpy.matlib
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
model = Word2Vec.load("data\word_model.mod")
#model = Word2Vec.load("data/word_model.mod")




count =0
document_name ="d04a/"
am ="d04aa"
dm ="FT923-5089"
doc2 = codecs.open('D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/prep1.txt', "w")
head = open("D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/head.txt")
prep = open("D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/prep2.txt")

mystopwords = ["one", "on", "also", "next", "ask", "set", "the", "for", "show" , "now", "need", "post" , "said"]


def prepdata():
    ps = PorterStemmer()
    # word_tokenize accepts a string as an input, not a file.
    stop_words = set(stopwords.words('english'))
    data = ' '
    headline = " "
    originallength = []
    # with open("D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/FT923-5089") as f:
    with open("D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/" + document_name + dm) as f:
        with open( "D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/head.txt","w") as f1:
            for line in f:
                data = data + line.replace('\n', ' ')
            #print(data)
            firstDelPos = data.find("<TEXT>")  # get the position of delimiter [
            secondDelPos = data.find("</TEXT>")  # get the position of delimiter ]
            extractedString = data[firstDelPos + 6:secondDelPos]  # get the string between two dels
            #print(extractedString)
            tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            doc2.write('\n'.join(tokenizer.tokenize(extractedString)))

           # print(headline)
            firstDelPos = data.find("<HEADLINE>")  # get the position of delimiter [
            secondDelPos = data.find("</HEADLINE>")  # get the position of delimiter ]
            extractedString = data[firstDelPos + 10:secondDelPos]  # get the string between two dels
            extractedString = extractedString.lower()
            words = extractedString.split(" ")
            stopdata = " "
            for r in words:
                if not r in stop_words:
                    if not r in mystopwords:
                        stopdata = stopdata + " " + r
            stopdata = re.sub(r'\([^)]*\)', '', stopdata)
            stopdata = re.sub('[^A-Za-z0-9\s]+', '', stopdata)
            #print(stopdata)
            f1.write(stopdata)
            f1.close()
            doc2.close()
            file1 = open("D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/prep1.txt")
            appendFile = open("D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/prep2.txt",'w')
            for line in file1:
                #print(line)
                line = line.lower()
                line = re.sub(r'\([^)]*\)', '', line)
                line = re.sub('[^A-Za-z0-9\s]+', '', line)
                words = line.split(" ")
                m = len(words)
                originallength.append(m)
                stopdata = " "
                for r in words:
                    if not r in stop_words:
                        if not r in mystopwords:
                            stopdata = stopdata + " " + r

                #print(stopdata)
                appendFile.write(stopdata)

            #print(originallength)
            print("original length",len(originallength))
            return originallength



def prepfeature():
    feat = open(        "D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/feature1.txt",         "w")
    arr1 = []
    for line2 in prep:
        #print(line2)
        arr1.append(line2)

    headline = []
    for line in head:
        #print(line)
        headline.append(line)

    finallist = []
    tfgram = 0
    line_no = 1

    def tfidfonegram(arr1, ngram):
        vectorizer = CountVectorizer(ngram_range=(ngram, ngram))
        X = vectorizer.fit_transform(arr1)
        vectorizer = TfidfVectorizer(ngram_range=(ngram, ngram))  # You can still specify n-grams here.
        X = vectorizer.fit_transform(arr1)
        vectorizer = TfidfVectorizer(ngram_range=(ngram, ngram), norm=None)  # You can still specify n-grams here.
        X = vectorizer.fit_transform(arr1)
        # print(X.toarray())
        xar = X.toarray()
        #print(xar)

        rows = len(xar)
        cols = len(xar[0])
        total = 0
        wo = []
        for m in range(0, rows):
            rowtotal = 0
            for n in range(0, cols):
                rowtotal = rowtotal + (xar[m][n])
            # print(rowtotal)
            wo.append(rowtotal)
        # print(wo[5])
        return wo


    # we used xx-1 as similarity returns 0 if 100% similar else returns 1 if disimilar
    def wmdsim(arr1, headline):
        wo = []
        for f2 in arr1:
            for i in headline:
                words = len(f2)

                vocabulary = [w for w in set(f2.lower().split() + i.lower().split()) if
                              w in model.wv.vocab ]

                vect = CountVectorizer(vocabulary=vocabulary).fit([i, f2])
                W_ = np.array([model[w] for w in vect.get_feature_names() if w in model])
                D_ = euclidean_distances(W_)
                D_ = D_.astype(np.double)
                D_ /= D_.max()  # just for comparison purposes
                v_1, v_2 = vect.transform([i, f2])
                v_1 = v_1.toarray().ravel()
                v_2 = v_2.toarray().ravel()
                v_1 = v_1.astype(np.double)
                v_2 = v_2.astype(np.double)
                v_1 /= v_1.sum()
                v_2 /= v_2.sum()
                # print("cosine(doc_1, doc_2) = {:.2f}".format(cosine(v_1, v_2)))
                xx = emd(v_1, v_2, D_)
                xx = 1 - xx
            wo.append(xx)
        # print(wo)
        return wo


    def cosinesim(arr1, headline):
        wo = []
        for f2 in arr1:
            for i in headline:
                words = len(f2)

                vocabulary = [w for w in set(f2.lower().split() + i.lower().split()) if
                              w in model.wv.vocab]

                vect = CountVectorizer(vocabulary=vocabulary).fit([i, f2])
                W_ = np.array([model[w] for w in vect.get_feature_names() if w in model])
                D_ = euclidean_distances(W_)
                D_ = D_.astype(np.double)
                D_ /= D_.max()  # just for comparison purposes
                v_1, v_2 = vect.transform([i, f2])
                v_1 = v_1.toarray().ravel()
                v_2 = v_2.toarray().ravel()
                # print("cosine(doc_1, doc_2) = {:.2f}".format(cosine(v_1, v_2)))
                xx = cosine(v_1, v_2)
                xx = 1 - xx
            wo.append(xx)
        # print(wo)
        return wo

    output = []
    ngram = 1
    output = tfidfonegram(arr1, ngram)
    ngram = 2
    output2 = tfidfonegram(arr1, ngram)
    ngram = 3
    output3 = tfidfonegram(arr1, ngram)
    cos = cosinesim(arr1, headline)
    wmdd = []
    wmdd = wmdsim(arr1, headline)

    #print(wmdd)
    # output.append(5)
    # output.append(tfidfonegram(arr1))

    # globle parameter

    for line in arr1:
        list = []
        wordsline = line.split()
        m = len(wordsline)
        list.append(m)
        list.append(output[tfgram])
        list.append(output2[tfgram])
        list.append(output3[tfgram])
        list.append(cos[tfgram])
        list.append(wmdd[tfgram])
        print("features value")
        print(list)
        a = np.asarray(list)
        # print(a)
        tfgram = tfgram + 1
        dip = str(list).replace("[", "")
        dip = dip.replace("]", "")
        #print(dip)
        feat.write(" " + dip)
        del list[:]
        feat.write('\n')





m = prepdata()
print(m)
prepfeature()
