"""
Youfei Zhang 

1. Answer: 

The "amountAlcohol" for each class seem to follow mGaussian distribution 
rather than Exponential.

"""

import scipy.io
varIn = scipy.io.loadmat('hw1data.mat')
trainData = varIn['trainData']
testData = varIn['testData']


# 2. learnMean

def learnMean(Data, classNum):
    """
    Data: a numpy array with shape/size[N, 2]
    classNum: a single integer
    output: a single number of the mean of this class 
    """
    # slice rows when column[0] = class label 
    sub = Data[Data[:,0] == classNum]
    
    # sum amountAlcohol by column, then slice to get sumAlc
    # average over the number of people in this class
    sumAlc = sub.sum(axis=0)[1]
    numPeople = len(sub)
    meanAlc = sumAlc/numPeople
    
    return meanAlc    
    


# 3. labelML

import math 
import numpy as np

def labelML(amountAlc, meanVector):
    """
    amountAlc: integer, an amountAlcohol measurement
    meanVector: a vector, the mean amountAlcohol values for the four shopper classes 
    output: a single character among {M, Y, A, or S} for the most likely class
    """

    # 1) caculate the variance of data: std = 2
    variance = 2 ** 2
    label = np.array(['M', 'Y', 'A', 'S'])
    dic = dict(zip(label, meanVector))
    
    # 2) get the PDF of each label 
    # 3) compare the PDF and return the largest one
    for key, value in dic.items():
        exponent = math.exp(-(math.pow(amountAlc - value, 2) / (2 * variance)))
        dic[key] = ((1/(math.sqrt(2 * math.pi * variance))) * exponent)
        label = max(dic, key=dic.get)
        
    return label
       


# 4. labelMP

import math 

def labelMP(amountAlc, meanVector, priorVector):
    """
    amountAlc: integer 
    meanVector: numpy array with shape/size [1,4]
    priorVector: numpy array
    output: a single character, M, Y, A, or S, for the most probable class
    """
    
    # 1. caculate the variance of data: std = 2
    # 2. get the PDF of each label 
    # 3. compare PDF and return the largest one
    
    variance = 2 ** 2
    label = np.array(['M', 'Y', 'A', 'S'])
    dicMean = dict(zip(label, meanVector))
    dicPrior = dict(zip(label, priorVector))

    dicPDF = {}
    for key, value in dicMean.items():
        exponent = math.exp(-(math.pow(amountAlc - value, 2) / (2 * variance)))
        result = ((1/(math.sqrt(2 * math.pi * variance))) * exponent)
        dicPDF[key] = result

    dicBayes = {}
    for key, value in dicPrior.items():  
        if key in dicPDF:
            dicBayes[key] = dicPrior[key] * dicPDF[key]

    label = max(dicBayes, key=dicBayes.get)
    
    return label


# 5. evaluateML

import pandas as pd 
    
def evaluateML(testData, meanVector):
    """
    testData: a numpy array of size/shape [N,2] 
    meanVector: the numpy array with shape/size [1,4] 
    output: the fraction of correctly-labeled data points in the test set [0,1]
    """
    
    # 1. predict the label for all the test data 
    # 2. compare the label of all the test data with their true label 
    # 3. calculate the % of correct labels 
    
    test = pd.DataFrame(testData).rename(index=str, columns={0:"label", 1:"amountAlcohol"})
    labelDic = {1: 'M', 2: 'Y', 3: 'A', 4: 'S'}
    test['label2'] = [labelDic[x] for x in test.label]

    predict = []
    for i in test['amountAlcohol']:
        predict.append(labelML(i, meanVector))
    test['predict'] = predict

    test['true'] = np.where((test['predict'] == test['label2']), 1, 0)
    accuracy = len(test[test['true'] == 1])/len(test['true'])
    
    return accuracy



# 6. evaluateMP

def evaluateMP(testData, meanVector, priorVector):
    """
    testData: a numpy array of size/shape [N,2] 
    meanVector: the numpy array with shape/size [1,4] 
    output: the fraction of correctly-labeled data points in the test set [0,1]
    """
    
    # 1. predict the label for all the test data 
    # 2. compare the label of all the test data with their true label 
    # 3. calculate the % of correct labels 
    
    test = pd.DataFrame(testData).rename(index=str, columns={0:"label", 1:"amountAlcohol"})
    labelDic = {1: 'M', 2: 'Y', 3: 'A', 4: 'S'}
    test['label2'] = [labelDic[x] for x in test.label]

    predict = []
    for i in test['amountAlcohol']:
        predict.append(labelMP(i, meanVector, priorVector))
    test['predict'] = predict

    test['true'] = np.where((test['predict'] == test['label2']), 1, 0)
    accuracy = len(test[test['true'] == 1])/len(test['true'])
    
    return accuracy  

"""
7. Report the percent of correctly labeled test data for 
max likelihood and max posterior separately when means are learned:

1) on the first 6 data points in the training set

MLE: 83.33% 
MAP: 50%

evaluateML(train6, mean6) = 0.8333333333333334
evaluateMP(train6, mean6, priorVector) = 0.5

2) means are learned on the 1st on the first 18 data points

MLE: 88.88% 
MAP: 77.77%

evaluateML(train18, mean18) = 0.8888888888888888
evaluateMP(train18, mean18, priorVector) = 0.7777777777777778


3) means are learned on the 1st on the first 54 data points

MLE: 64.81% 
MAP: 64.81%

evaluateML(train54, mean54) = 0.6481481481481481
evaluateMP(train54, mean54, priorVector) = 0.6481481481481481

4) means are learned on the 1st on the first 162 data points

MLE: 59.26% 
MAP: 61.72%

evaluateML(train162, mean162): 0.5925925925925926
evaluateMP(train54, mean54, priorVector): 0.6172839506172839

"""

# 8. labelMP2

import scipy.io

q8 = scipy.io.loadmat('hw1dataQ8.mat')
train8 = q8['trainData']
test8 = q8['testData']

def labelMP2(amountDrinks, meansMatrix, priorVector):
    """
    amountDrinks: a numpy array with shape/size [1,2] contains [amtAlcohol, amtSoda]
    meansMatrix: a numpy array with shape/size [2,4]
    priorVector: numpy array
    output: a single character, M, Y, A, or S, for the most probable class
    """
    
    # 1. caculate the variance of data: std = 2
    # 2. get the PDF of each label of each feature
    # 3. pdf * prior
    
    variance = 2 ** 2
    label = np.array(['M', 'Y', 'A', 'S'])
    mean = pd.DataFrame(meansMatrix).rename(index=str, columns={0:"M", 1:"Y", 2:"A", 3:"S"}).transpose()

    dicPDFAlc = []
    for i in mean['0']:
        exponent = math.exp(-(math.pow(amountDrinks[0] - i, 2) / (2 * variance)))
        result = ((1/(math.sqrt(2 * math.pi * variance))) * exponent)
        dicPDFAlc.append(result)
    mean['pdfAlc'] = dicPDFAlc

    dicPDFSod = []
    for i in mean['1']:
        exponent = math.exp(-(math.pow(amountDrinks[1] - i, 2) / (2 * variance)))
        result = ((1/(math.sqrt(2 * math.pi * variance))) * exponent)
        dicPDFSod.append(result)
    mean['pdfSod'] = dicPDFSod
    
    mean['pior'] = priorVector
    mean['class'] = mean['pior'] * mean['pdfAlc'] * mean['pdfSod']

    label = mean[mean['class'] == max(mean['class'])].index[0]
    
    return label

