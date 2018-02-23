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
from keras.utils.np_utils import to_categorical
from keras.layers import GlobalAveragePooling1D,MaxPooling1D,Flatten
from keras import backend as K
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
max_features = 600000
maxlen = 20000  # cut texts after this number of words (among top max_features most common words)
batch_size = 32
embedding_dims = 256
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2

x_train = np.load("/home/deeplearning2/ai/data/train.npy")
x_test = np.load("/home/deeplearning2/ai/data/test.npy")
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

label = pd.read_csv("/home/deeplearning2/ai/data/money_label.csv")
y_train = to_categorical(label.as_matrix(), num_classes=9)
trainrow=pd.read_csv("/home/deeplearning2/ai/data/trainrow.csv")
testrow = pd.read_csv("/home/deeplearning2/ai/data/testrow.csv")

model = Sequential()
model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
model.add(Dropout(0.15))
model.add(Conv1D(250,3,padding='valid',activation='relu',strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(250))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(9))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(np.concatenate([x_train[0:30000],x_train[60000:90000]],axis=0),np.concatenate([y_train[0:30000],y_train[60000:90000]],batch_size=32,epochs=1)
model.fit(x_train[30000:90000], y_train[30000:90000],batch_size=32,epochs=1)

a=model.predict(x_train[0:30000])
#model.fit(X_train, Y_train,batch_size=32,epochs=3,validation_data=(X_test, Y_test))
weights=[]
for i in range(len(model.layers)):
    weights.append(model.layers[i].get_weights())

K.clear_session()

#得到输出特征
model = Sequential()
model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
model.add(Dropout(0.15))
model.add(Conv1D(250,3,padding='valid',activation='relu',strides=1))
model.add(GlobalMaxPooling1D())
for i in range(len(model.layers)):
    model.layers[i].set_weights(weights[i])

trainmf = model.predict(x_train)
testmf = model.predict(x_test)

a=np.load('/home/deeplearning2/ai/code/mtrain1.npy')
b=np.load('/home/deeplearning2/ai/code/mtest1.npy')
trainf=np.concatenate([trainmf,a],axis=1)
testf=np.concatenate([testmf,b],axis=1)
'''
#附加特征
trainappd=[]
file = open("/home/deeplearning2/ai/code/trainjine.txt",encoding='utf-8')
for line in file:
	trainappd.append([float(line[0:-1])])

i=0
file = open("/home/deeplearning2/ai/code/trainjinemax.txt",encoding='utf-8')
for line in file:
	trainappd[i].append(float(line[0:-1]))
	i=i+1

trainappd = np.array(trainappd)
trainmf=np.concatenate([trainmf,trainappd],axis=1)

testappd=[]
file = open("/home/deeplearning2/ai/code/testjine.txt",encoding='utf-8')
for line in file:
    testappd.append([float(line[0:-1])])

i=0
file = open("/home/deeplearning2/ai/code/testjinemax.txt",encoding='utf-8')
for line in file:
    testappd[i].append(float(line[0:-1]))
    i=i+1

testappd = np.array(testappd)
testmf=np.concatenate([testmf,testappd],axis=1)
'''

#xgb训练
import xgboost as xgb
params = {
            'objective': 'multi:softmax',
            'eta': 0.05,
            'max_depth': 12,
            'eval_metric': 'merror',
            'num_class':9,
            'missing': 0,
            'silent' : 1,
            'nthread':10
            }

c=label.as_matrix()
xgbtrain = xgb.DMatrix(trainf,c)
xgbtest = xgb.DMatrix(testf)
watchlist = [ (xgbtrain,'train'), (xgbtrain, 'test') ]
num_rounds=200
model = xgb.train(params, xgbtrain, num_rounds, watchlist, early_stopping_rounds=200)

ans=[]
probas=model.predict(xgbtest)
for i in range(len(probas)):
    ans.append(probas[i])

ansD = pd.DataFrame(ans,columns=['money'],index=None)
ansD.to_csv("../money-12-4.csv",index=None)











