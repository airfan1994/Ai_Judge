#!/usr/bin/python
#coding:utf-8
import time
import sys
import string
import numpy as np
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
reload(sys)
sys.setdefaultencoding('utf-8')

i=1
if i>0:
        train_file_name = "trainraw2.txt"
        label1 = pd.read_csv('label1.txt',header=None)
        test_file_name = "testraw2.txt"
        traindata = []
        testdata = []
        for line in open(train_file_name,'r').readlines():#è¯»å–åˆ†ç±»åºåˆ—
                traindata.append(line.strip())
        for line in open(test_file_name,'r').readlines():#è¯»å–åˆ†ç±»åºåˆ—
                testdata.append(line.strip())
        alldata = traindata+testdata
        count_v1 = TfidfVectorizer(max_df=10000,min_df=6)
        idftransformer = count_v1.fit(alldata)
        trainvec = idftransformer.transform(traindata)
        testvec = idftransformer.transform(testdata)
        clf=LinearSVC()
        clf.fit(trainvec,label1)
        label2 = clf.predict(testvec)
        df = pd.DataFrame(label2)
        df.to_csv("predictidf.csv",index=False,header=None)

