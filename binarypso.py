from rouge import rouge
import nltk.data
import codecs
import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction import _stop_words
import random
import numpy as np
import random as rn
import numpy as np
import matplotlib.pyplot as plt

#from nsgacrowdingdis import *
from random import randrange, uniform
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

#model = Word2Vec.load("data/word_model.mod")

import numpy.matlib
import numpy as np

model = Word2Vec.load("data\word_model.mod")




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

    
        # print(wo)
        return wo


    def cosinesim(arr1, headline):
        wo = []
       
    output3 = tfidfonegram(arr1, ngram)
    cos = cosinesim(arr1, headline)
    wmdd = []
    wmdd = wmdsim(arr1, headline)

    #print(wmdd)
    # output.append(5)
    # output.append(tfidfonegram(arr1))

    # globle parameter

    for line in arr1:
       
        feat.write('\n')


def obj1():
    
    return feature_scoref1


def obj2():
    

    print(wmdd)
    return wmdd,count


def create_starting_population(individuals, chromosome_length):
        

        return population


def create_reference_solution(chromosome_length):
    

    return reference


def populationlength(population):
        
        return newpopulation


def objective1(initial_population, xf):
       

    return population


def lenghi(cplen):
    list = []
    indexi = []
    sumpop = 0
    sumf1 = 0
    mm = 0
    for item in range(0, chromosome_length-1):
        itea = int(cplen[item])

        if cplen[item] != 0:
            sumpop = sumpop + m[mm]
            mm = mm + 1
    if (sumpop > summarylimit) or (sumpop < 50):
            #if sumpop < 50:
        sumpop = 0

        #print(list)

    #print(newpopulation)
    return sumpop



def objective11(inip, xf):
    


def mr(cm):
    number_of_ones = int(cm /2)

    # Build an array with an equal mix of zero and ones
    reference = np.zeros(cm)
    reference[0: number_of_ones] = 1

    # Shuffle the array to mix the zeros and ones
    np.random.shuffle(reference)

    return reference


def tmo(npi , newdata):
   rtedvalue[0])
    for eachpop in range(0, cm-1):
        if (((aresum[eachpop][0]) == sortedvalue[0])):
            #print("found match",eachpop)
            x = tmdata[eachpop]

    #print(x)
    return x, sortedvalue[0]

    #print(x)



def rev(a,svc,smc):
    if smc + svc > chromosome_length:
        smc = (chromosome_length - smc) -1
    a[svc:smc] = a[svc:smc][::-1]
    return a





def childgeneration(nm, cpbest, cpbestvalues, gbest, gbestvalue):
   
# Create reference solution
# (this is used just to illustrate GAs)
reference = create_reference_solution(chromosome_length)
#print("ref",reference)

# Create starting population
population = create_starting_population(population_size, chromosome_length)
#print("initial population", population)

initial_population = populationlength(population)
cm = 0
for we in initial_population:
    cm =cm + 1

#print("no of population",cm)


objective1_score = objective1(initial_population, xf)
#print(objective1_score)
objective2_score = ovjective2(initial_population, wmdsim)
#print(objective2_score)
arr = np.concatenate((objective1_score, objective2_score), axis=1)

currentpbest = initial_population
aresum = np.sum((objective1_score, objective2_score), axis=0)
currentpbestvalues = aresum
newdata = []
print("current pbest population", currentpbest)
print("current pbest population objective score", currentpbestvalues)
plt.scatter(aresum,aresum)
plt.show()

#print(aresum)
sortedvalue = np.sort(aresum,axis=0)[::-1]
for eachpop in range (0,cm - 1):
    if(((aresum[eachpop])==sortedvalue[0])):
        print("gbest",aresum[eachpop],initial_population[eachpop])
        newdata.append(initial_population[eachpop])


gbest=newdata[0]
gbestvalue = sortedvalue[0]
print("gbest =",newdata[0])
population1 = initial_population
npopulation = population1
for generation in range(maximum_generation):
    print(" generation ", generation)
    #print("cm =", cm)
    for i in range(0, cm - 1):

        nm = population1[i]
        newi,pbesti,currentpbestvalue = childgeneration(nm,currentpbest[i],currentpbestvalues[i],gbest,gbestvalue,)
        x,newval = tmo(nm, newdata)
        population1[i,:]=newi
        currentpbestvalues[i,:]=currentpbestvalue
        currentpbest[i, :]= pbesti

        print("current pbest",newval)
        print("new tm updated data",population1[i])
        if(gbestvalue<currentpbestvalue):
            gbest = pbesti
            gbestvalue=currentpbestvalue



    for i in gbest:
        print("newdata",i)
m = []
print("best pop", gbest)
for i in gbest:
    m.append(i)

print(m)
prepp = open("D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/prep1.txt","r")
op = open("D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/op.txt","w")
x = m

iv = 0
print("optimal cat",m[iv])
output = " "
for line in prepp:
    if x[iv].__eq__(1.0):
        output = output + line.replace('\n', '.')
    iv = iv + 1

print("obtained output",output)
op.write(output.lower())

op.close()


data = []
s = " "
i = 0
extractedString  =" "
f2 = open("D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/orisum.txt",'w')
with open("D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/original.summaries/"+ am +"/perdocs") as f1:
#with open("D:/Dipanwita docs/cat swarm/summaries/summaries/" +am+"/perdocs") as f1:
    for line in f1:
        i = i + 1
        data.append(line)




for j in range (0, i-1):
    if data[j].__contains__(dm):
        for k in range (j+3,i-1):
            extractedString = extractedString + data[k]
            if data[k].__contains__("</SUM>"):
                break
        break

print("Actual summary",extractedString)

extractedString = extractedString.replace("\n","")
extractedString =extractedString.lower()
f2.write(extractedString)

plt.scatter(aresum,aresum)
plt.show()


f2.close()

f1 = open("D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/op.txt")
f2 = open("D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/orisum.txt")
#f4 = open("D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/archive.csv", "a")
#f3 = open("D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/archive.txt", "a")
f4 = open("D:/Dipanwita docs/phd/output/csowithdepsocsoGA.csv", "a")
f3 = open("D:/Dipanwita docs/phd/output/csowithdepsocsoga.txt", "a")

#f2 = open("D:/Dipanwita docs/scientific document/scisummnet_release1.1__20190413/top1000_complete/1aoutput/actualsumm.txt","r")
#f1 = open("D:/Dipanwita docs/scientific document/scisummnet_release1.1__20190413/top1000_complete/1aoutput/obtainedsumm.txt","r")
#f4 = open("D:/Dipanwita docs/scientific document/scisummnet_release1.1__20190413/top1000_complete/1aoutput/output.csv", "a")
#f3 = open("D:/Dipanwita docs/scientific document/scisummnet_release1.1__20190413/top1000_complete/1aoutput/outputtxt.txt","a")
#from gasumm import *
#from cd1fea import document_name



arr1 = ""
for line2 in f2:
    arr1 = line2


arr2 = " "
for line in f1:
    arr2 = line



li = "A00-1031"

#scores = Rouge.get_scores(arr1,arr)
scores =rouge.rouge_n_sentence_level(arr2, arr1, 1)
print("rough 1: " , scores)
li = li + ","+str(scores)+","
scores =rouge.rouge_n_sentence_level(arr2, arr1, 2)
print("rough 2: ", scores)
li = li + str(scores)
li = li.replace("=","")
li = li.replace("(","")
li = li.replace(")","")

li = li.replace("RougeScorerecall","")
li = li.replace("f1_measure","")
li = li.replace("precision","")
m = li
li =   li + "\t ORIGINAL TEXT" + arr1 +"\t\t  SYSTEM GENERATED OUTPUT" +arr2

print(li)
f3.write(li)
f3.write("\n")
f4.write(m)
f4.write("\n")








































