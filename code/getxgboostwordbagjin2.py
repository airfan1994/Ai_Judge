#!/usr/bin/python
#coding:utf-8
import time
import sys
import string
import numpy as np
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from xgboost import XGBClassifier
reload(sys)
sys.setdefaultencoding('utf-8')

i=1
if i>0:
	trainvecpart1 = pd.read_csv("trainvectotol.txt",header=None,sep=",")
	trainvecpart2 = pd.read_csv("trainjin.txt",header=None,sep=",")
	trainvecpart3 = pd.read_csv("trainlaw.txt",header=None,sep=",")
	label2 = pd.read_csv("label1.txt",header=None,sep=",")
	testvecpart1 = pd.read_csv("testvec.txt",header=None,sep=",")
	testvecpart2 = pd.read_csv("testjin.txt",header=None,sep=",")
	testvecpart3 = pd.read_csv("testlaw.txt",header=None,sep=",")
        trainvec1 = np.hstack((trainvecpart1,trainvecpart2))
        trainvec = np.hstack((trainvec1,trainvecpart3))
        testvec1 = np.hstack((testvecpart1,testvecpart2))
        testvec = np.hstack((testvec1,testvecpart3))
	#params = {'max_depth': 15,
        #      'eta': 0.1,
        #      'objective': 'multi:softmax',
        #      'num_class':len(rows),
        #      'silent':1,
	#      'nthread':8,
	#      'num_round':500
        #      }
        params1 = {
                'objective': 'multi:softmax',
                'eta': 0.05,
                'max_depth': 15,
                'eval_metric': 'merror',
                'seed': 0,
                'missing': 0,
                'num_class':9,
                'silent' : 1,
                 'nthread':10
                }
 #       params2 = {
#
     #           'objective': 'binary:logistic',
    #            'eta': 0.1,
    #            'max_depth': 7,
    ##            'eval_metric': 'logloss',
    #            'seed': 0,
    #            'missing': 0,
    #            'silent' : 1,
    #            'nthread':10
    #            }

    #    xgbtrain1 = xgb.DMatrix(trainvec, label1)
        xgbtrain2 = xgb.DMatrix(trainvec, label2)
        xgbtest = xgb.DMatrix(testvec)
   #     watchlist1 = [ (xgbtrain1,'train'), (xgbtrain1, 'test') ]
        watchlist2 = [ (xgbtrain2,'train'), (xgbtrain2, 'test') ]
        num_rounds=1000
#	model1 = xgb.train(params1, xgbtrain1, num_rounds, watchlist1, early_stopping_rounds=15)
	model2 = xgb.train(params1, xgbtrain2, num_rounds, watchlist2, early_stopping_rounds=15)
#	test_label1 = model1.predict(xgbtest)
 #       test_label1.to_csv(mallid+"/"+mallid+"_predict1.csv",index=False)

	test_label2 = model2.predict(xgbtest)
        testp = pd.DataFrame(test_label2)
        testp.to_csv("predict1_xgbwordbagjinlaw.csv",index=False)

