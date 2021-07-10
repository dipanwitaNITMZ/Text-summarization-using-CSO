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

   




m = prepdata()
print(m)
prepfeature()
