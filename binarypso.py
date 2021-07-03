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


def obj1():
    X = np.loadtxt( "D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/feature1.txt",  delimiter=",")
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

    # Scaled feature
    x_after_min_max_scaler = min_max_scaler.fit_transform(X)

    # print("\nAfter min max Scaling : \n", x_after_min_max_scaler)
    # print("\nSum of arr (keepdimension is True): \n", np.sum(x_after_min_max_scaler, axis = 1, keepdims = True))

    feature_scoref1 = np.sum(x_after_min_max_scaler, axis=1)
    #print(feature_scoref1)
    return feature_scoref1


def obj2():
    prep = open(
        "D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/prep2.txt")
    feat = open(
        "D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/feature2.txt",
        "w")

    count = 0
    arr1 = []
    for line2 in prep:
        count = count + 1
        arr1.append(line2)

    # print(count)

    # print(arr1)

    headline = arr1
    # print(headline)

    no = 0

    print(count)

    def wmdsim(arr1, headline):
        no = 0
        noo = 0
        wo = numpy.zeros(shape=(count, count))
        # print(wo[1][1])
        nx = 0
        for ele in range(0, count):
            f2 = arr1[ele]
            xn = 0
            for ile in range(0, count):
                i = headline[ile]
                no = no + 1
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
                v_1 = v_1.astype(np.double)
                v_2 = v_2.astype(np.double)
                v_1 /= v_1.sum()
                v_2 /= v_2.sum()

                # print("cosine(doc_1, doc_2) = {:.2f}".format(cosine(v_1, v_2)))
                xx = emd(v_1, v_2, D_)
                if (xx == "nun"):
                    xx = 0
                xx = 1 - xx
                if (xx >= 0.45):
                    xx = 0.0
                else:
                    xx = 1.0
                wo[ele][ile] = xx
                # print(ele, ile)
                # print(wo[ele][ile])
                xn = xn + 1
                # wo.append(xx)
            nx = nx + 1
            # print(wo)
        feat.write(" " + str(wo))
        feat.write('\n')
        return wo

    wmdd = wmdsim(arr1, headline)
    wmdd[np.diag_indices_from(wmdd)] = 0

    print(wmdd)
    return wmdd,count


def create_starting_population(individuals, chromosome_length):
        # Set up an initial array of all zeros
        population = np.zeros((individuals, chromosome_length))
        # Loop through each row (individual)
        for i in range(individuals):
            # Choose a random number of ones to create
            ones = random.randint(0, chromosome_length)
            # Change the required number of zeros to ones
            population[i, 0:ones] = 1
            # Sfuffle row
            np.random.shuffle(population[i])

        return population


def create_reference_solution(chromosome_length):
    print(chromosome_length)
    number_of_ones = int(chromosome_length - 1)
    reference = np.zeros(chromosome_length)
    reference[0: number_of_ones] = 1

        # Shuffle the array to mix the zeros and ones
    np.random.shuffle(reference)

    return reference


def populationlength(population):
        list = []
        indexi = []
        for i in range(0, individuals - 1):
            cp = population[i]
            sumpop = 0
            sumf1 = 0
            mm = 0
            for item in cp:
                if item != 0:
                    sumpop = sumpop + m[mm]

                mm = mm + 1
            if (sumpop > summarylimit) or (sumpop < 50):
                # if sumpop < 50:
                sumpop = 0
                indexi.append(i)
                # print(indexi)

            list.append(sumpop)
            # print(list)
        list.sort()

        newpopulation = np.delete(population, indexi, axis=0)
        # print(newpopulation)
        return newpopulation


def objective1(initial_population, xf):
        list = []
        indexi = []
        xx = 0
        v = 0
        for ea in initial_population:
            xx = xx + 1
        # print(xx)
        count = 0
        for i in range(0, xx - 1):
            cp = initial_population[i]
            sum = 0
            for cpo in range(0, chromosome_length - 1):
                sum = sum + (cp[cpo] * xf[cpo])
            indexi.append(sum)

            # print(initial_population[i])
            # newvalue = np.array(initial_population[i] * np.array(xf))
            # print(newvalue)
            # cp = initial_population[i]
            # obj1 = np.sum(newvalue, axis=1, keepdims=True)
            # print(obj1)

            count = count + 1
        # print(count)
        # print(indexi)
        # print(indexi.shape)
        arrop = numpy.array(indexi)
        arrop = arrop.reshape((-1, 1))
        # print(arrop.shape)
        return arrop


def ovjective2(initial_population, wmdsim):
        list = []
        indexi = []
        xx = 0
        for ea in initial_population:
            xx = xx + 1
        # print(xx)
        popum = 0
        count = 0
        for i in range(0, xx - 1):
            cp = initial_population[i]
            # print("hghjgj", cp[1])

            # print(wmdd)
            mm = 0

            for j in range(0, chromosome_length - 1):
                m = 0
                for k in range(0, chromosome_length - 1):
                    m = m + (wmdsim[j][k] * cp[k])
                    # print(m)
                mm = mm + (cp[j] * m)
                # print(mm)

            count = count + 1

            popum = popum + 1
            # print( popum)
            indexi.append(mm)
        # print(count)
        # print(indexi)

        arrop = numpy.array(indexi)
        arrop = arrop.reshape((-1, 1))
        # print(arrop.shape)
        # print(arrop)
        return arrop



def breed_by_crossover(parent_1, parent_2):
    """
    Combine two parent chromsomes by crossover to produce two children.
    """
    # Get length of chromosome
    chromosome_length = len(parent_1)

    cross = []

    # Pick crossover point, avoding ends of chromsome
    crossover_point = rn.randint(1, chromosome_length - 1)

    # Create children. np.hstack joins two arrays
    child_1 = np.hstack((parent_1[0:crossover_point],
                         parent_2[crossover_point:]))

    child_2 = np.hstack((parent_2[0:crossover_point],
                         parent_1[crossover_point:]))
    cross.append(child_1)
    cross.append(child_2)

    # Return children
    return child_1,child_2


def randomly_mutate_population(population, mutation_probability):
    """
    Randomly mutate population with a given individual gene mutation
    probability. Individual gene may switch between 0/1.
    """
    # Apply random mutation
    random_mutation_array = np.random.random(size=(population.shape))

    random_mutation_boolean = \
        random_mutation_array <= mutation_probability

    population[random_mutation_boolean] = \
        np.logical_not(population[random_mutation_boolean])

    # Return mutation population
    return population


def breed_population(population):
    """
    Create child population by repetedly calling breeding function (two parents
    producing two children), applying genetic mutation to the child population,
    combining parent and child population, and removing duplicatee chromosomes.
    """
    # Create an empty list for new population
    new_population = []
    population_size = population.shape[0]
    # Create new popualtion generating two children at a time
    for i in range(int(population_size / 2)):
        parent_1 = population[rn.randint(0, population_size - 1)]
        parent_2 = population[rn.randint(0, population_size - 1)]
        child_1, child_2 = breed_by_crossover(parent_1, parent_2)
        new_population.append(child_1)
        new_population.append(child_2)

    # Add the child population to the parent population
    # In this method we allow parents and children to compete to be kept
    population = np.vstack((population, np.array(new_population)))
    population = np.unique(population, axis=0)

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
    list = []
    indexi =[]
    xx = 0
    v = 0
    for eaob1 in inip:
        xx = xx + 1
    print(xx)
    count = 0
    leng = 0
    for iob1 in range(0, xx-1):
        cp = inip[iob1]
        leng = lenghi(cp)
        #print(leng)
        #print(leng)
        sums = 0
        for cpo in range (0, chromosome_length-1):
            sums = sums + (cp[cpo] *xf[cpo])

        if leng == 0:
            sums = 0
        sums = float(sums)
        indexi.append(sums)

        # print(initial_population[i])
        # newvalue = np.array(initial_population[i] * np.array(xf))
        # print(newvalue)
        # cp = initial_population[i]
        # obj1 = np.sum(newvalue, axis=1, keepdims=True)
        # print(obj1)

        count = count + 1
        # print(count)
        # print(indexi)
        # print(indexi.shape)
    arrop = numpy.array(indexi)
    arrop = arrop.reshape((-1, 1))
    # print(arrop.shape)
    return arrop




def ovjective22(popobj, wmdsim):
    list = []
    indexi = []
    xx = 0
    for ea in popobj:
        xx = xx + 1
    # print(xx)
    popum = 0
    count = 0
    for i in range(0, xx - 1):
        cp = popobj[i]
        leng = lenghi(cp)
        #print("hghjgj", cp[1])

        #print(wmdd)
        mm= 0

        for j in range (0, chromosome_length -1):
            m = 0
            for k in range(0, chromosome_length -1):
                m = m + (wmdsim[j][k] * cp[k])
                #print(m)
            mm = mm + (cp[j] * m)
            #print(mm)

        count = count + 1

        popum = popum + 1
        if leng == 0:
            mm = 0
        #print( popum)
        indexi.append(mm)
    #print(count)
    #print(indexi)

    arrop = numpy.array(indexi)
    arrop = arrop.reshape((-1, 1))
    #print(arrop.shape)
    #print(arrop)
    return arrop



def identify_pareto(scores, population_ids):
    """
    Identifies a single Pareto front, and returns the population IDs of
    the selected solutions.
    """
    population_size = scores.shape[0]
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]



def mr(cm):
    number_of_ones = int(cm /2)

    # Build an array with an equal mix of zero and ones
    reference = np.zeros(cm)
    reference[0: number_of_ones] = 1

    # Shuffle the array to mix the zeros and ones
    np.random.shuffle(reference)

    return reference


def tmo(npi , newdata):
    x = 0
    tmdata = []

    for valuezz in newdata:
        c1,c2 =breed_by_crossover(valuezz, npi)
        tmdata.append(c1)

        tmdata.append(c2)

    #tmdata = numpy.array(tmdata)
    tmdata.append(npi)
    cm = 0
    x = 0
    for ele in tmdata:
        cm = cm + 1
    #print(c1)

    objective_score1 = objective11(tmdata, xf)
    #print(objective1_score)
    objective_score2 = ovjective22(tmdata, wmdsim)
    #print(objective2_score)

    arr = np.concatenate((objective_score1, objective_score2), axis=1)
    aresum = np.sum((objective1_score, objective2_score), axis=0)


    li =0
    #for em in aresum:
    #    print("em",em)
    sortedvalue = np.sort(aresum, axis=0)[::-1]
    #print("sorted value",sortedvalue[0])
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
    x = 0
    tmdata = []
    c7 = random.shuffle(nm)
    li=0

    a = nm
    #a is a random vector
    random.shuffle(a)
    #c3 = [nm ^ c1 for nm, c1 in zip(nm, c1)]
    #c3 = [nm[index] ^ c1[index] for index in range(len(nm))]
    #c4 = [a[index] & c3[index] for index in range(len(a))]

    random.shuffle(a)
    #c5 = [nm[index] ^ c2[index] for index in range(len(nm))]
    #c6 = [a[index] & c5[index] for index in range(len(a))]

    # finally or
    #c7 = [c4[index] | c6[index] for index in range(len(c4))]
    tmdata.append(nm)
    tmdata.append(c7)
    objective_score1 = objective11(tmdata, xf)
    #print(objective1_score)
    objective_score2 = ovjective22(tmdata, wmdsim)
    # print(objective2_score)
    arr = np.concatenate((objective_score1, objective_score2), axis=1)
    aresum = np.sum((objective1_score, objective2_score), axis=0)
    aresum = numpy.array(aresum)
    aresum= aresum.reshape((-1, 1))
    for em in tmdata:
        li = li + 1
    print("number:",li)
    sortedvalue = np.sort(aresum, axis=0)[::-1]
    #print(sortedvalue)
    for eachpop in range(0, li-1):
        if (((aresum[eachpop]) == sortedvalue[0])):
            #print(aresum[eachpop], initial_population[eachpop])
            cpbest =tmdata[eachpop]
            #pbesti=newi
            cpbestvalues = sortedvalue[0]
            break

    #print(x)
    return a, cpbest,cpbestvalues



m = prepdata()
prepfeature()
fc = obj1()
wmdd,count = obj2()



xf = fc

wmdsim = wmdd

k = 100
chromosome_length = count

fs = [1, 2, 3, 5, 4, 6]
individuals = chromosome_length * 5
population_size = (chromosome_length * 5)
maximum_generation = 2
best_score_progress = []
summarylimit = 110

# Tracks progress


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








































