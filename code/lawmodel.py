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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
max_features = 600000
maxlen = 15000  # cut texts after this number of words (among top max_features most common words)
batch_size = 32
embedding_dims = 256
filters = 250
kernel_size = 3
hidden_dims = 250


x_train = np.load("/home/deeplearning2/ai/data/train.npy")
x_test = np.load("/home/deeplearning2/ai/data/test.npy")
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

label = pd.read_csv("/home/deeplearning2/ai/data/law_label.csv")

lawset = set(label['label'])
trainrow=pd.read_csv("/home/deeplearning2/ai/data/trainrow.csv")
testrow = pd.read_csv("/home/deeplearning2/ai/data/testrow.csv")




#lawlist=[12,17,18,19,22,30,31,36,37,38,41,43,55,56,57,58]
for law in lawset:   
    pos_row = set(label[label['label']==law]['row_id'])
    if len(pos_row)>1000: 
        print(law)
        hehe=pd.DataFrame()
        hehe['lab'] = trainrow['row_id'].isin(pos_row)
        hehe['lab'] = hehe['lab'].apply(lambda x: 1 if x else 0)
        y_train=hehe['lab'].as_matrix()
        X_train, Y_train = shuffle(x_train, y_train)

        model = Sequential()
        model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
        model.add(Dropout(0.15))
        model.add(Conv1D(256,3,padding='valid',activation='relu',strides=1))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(250))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        model.fit(X_train, Y_train,batch_size=batch_size,epochs=1) 
        weights=[]
        for i in range(len(model.layers)):
            weights.append(model.layers[i].get_weights())

        K.clear_session()

        model = Sequential()
        model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
        model.add(Dropout(0.15))
        model.add(Conv1D(256,3,padding='valid',activation='relu',strides=1))
        model.add(GlobalMaxPooling1D())
        for i in range(len(model.layers)):
            model.layers[i].set_weights(weights[i])
        

        trainmf=model.predict(x_train)
        testmf=model.predict(x_test)
        np.save('/home/deeplearning2/ai/feature/train_'+str(law),trainmf)
        np.save('/home/deeplearning2/ai/feature/label_'+str(law),y_train)
        np.save('/home/deeplearning2/ai/feature/test_'+str(law),testmf)
        K.clear_session()
        '''
        import xgboost as xgb
        params = {
                    'objective': 'binary:logistic',
                    'eta': 0.05,
                    'max_depth': 12,
                    'eval_metric': 'logloss',
                    'missing': 0,
                    'silent' : 1,
                    'nthread':10
                    }

        xgbtrain = xgb.DMatrix(trainmf, y_train)
        xgbtest = xgb.DMatrix(testmf)
        num_rounds=150
        model = xgb.train(params, xgbtrain, num_rounds)
        ans=[]
        probas=model.predict(xgbtest)
        for i in range(len(probas)):
            if probas[i]>=0.5:
                  ans.append([testrow.iloc[i]['row_id'],law])

        ansD=pd.DataFrame(ans,columns=['row_id','law'])
        ansD.to_csv("/home/deeplearning2/ai/re2/"+str(law)+".csv",index=None)
        '''
        
