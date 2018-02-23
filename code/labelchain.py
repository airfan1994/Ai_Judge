#coding: utf-8
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
from sklearn.utils import shuffle
from keras.layers import GlobalAveragePooling1D
from keras import backend as K
import random
import os
from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
max_features = 600000
maxlen = 15000  # cut texts after this number of words (among top max_features most common words)
batch_size = 32
embedding_dims = 256
filters = 250
kernel_size = 3
hidden_dims = 250


label = pd.read_csv("/home/deeplearning2/ai/data/law_label.csv")
lawset = set(label['label'])
lawlist=[]
for law in lawset:
	if(len(set(label[label['label']==law]['row_id']))>=100):
		lawlist.append(law)

trainrow=pd.read_csv("/home/deeplearning2/ai/data/trainrow.csv")
testrow = pd.read_csv("/home/deeplearning2/ai/data/testrow.csv")
trainfeatureDict={}
testfeatureDict={}
ans={}

for law in lawlist:
	trainfeatureDict[law]=np.load('/home/deeplearning2/ai/feature/train_'+str(law)+".npy")
	testfeatureDict[law]=np.load('/home/deeplearning2/ai/feature/test_'+str(law)+".npy")
	ans[law]=[]



for i in range(3):
	if i>0:
		random.shuffle(lawlist)
	trainarray=[]
	testlist=[]
	for j in range(len(lawlist)):
		law=lawlist[j]
		print(law)
		y_train=np.load('/home/deeplearning2/ai/feature/label_'+str(law)+".npy")
		trainf = trainfeatureDict[law]
		testf = testfeatureDict[law]
		if j>0:
			trainf=np.concatenate([trainf,trainarray],axis=1)
			testf=np.concatenate([testf,np.array(testlist)],axis=1)
		import xgboost as xgb
		params = {'objective': 'binary:logistic','eta': 0.05,'max_depth': 12,'eval_metric': 'logloss','missing': 0,'silent' : 1,'nthread':10}
        xgbtrain = xgb.DMatrix(trainf, y_train)
        xgbtest = xgb.DMatrix(testf)
        num_rounds=70
        watchlist = [ (xgbtrain,'train'), (xgbtrain, 'test') ]
        model = xgb.train(params, xgbtrain, num_rounds,watchlist, early_stopping_rounds=100)
		a=model.predict(xgbtest)
		for k in range(len(a)):
			if a[k]>=0.5:
				ans[law].append([testrow.iloc[k]['row_id'],law,1])
				if j==0:
					testlist.append([1])
				else:
					testlist[k].append(1)
			else:
				ans[law].append([testrow.iloc[k]['row_id'],law,0])
				if j==0:
					testlist.append([0])
				else:
					testlist[k].append(0)
		if j==0:
			trainarray=y_train.reshape(len(y_train),1)
		else:
			trainarray=np.concatenate([trainarray,y_train.reshape(len(y_train),1)],axis=1)







	
