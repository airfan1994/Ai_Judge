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
maxlen = 24100  # cut texts after this number of words (among top max_features most common words)
batch_size = 32
embedding_dims = 290
filters = 250
kernel_size = 3
hidden_dims = 250




x_train = np.load("/home/deeplearning2/ai/data/train.npy")
x_test = np.load("/home/deeplearning2/ai/data/test.npy")
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

label = pd.read_csv("/home/deeplearning2/ai/data/label.csv")

lawset = set(label['label'])
trainrow=pd.read_csv("/home/deeplearning2/ai/data/trainrow.csv")
testrow = pd.read_csv("/home/deeplearning2/ai/data/testrow.csv")


law=67
pos_row = set(label[label['label']==law]['row_id'])
hehe=pd.DataFrame()
hehe['lab'] = trainrow['row_id'].isin(pos_row)
hehe['lab'] = hehe['lab'].apply(lambda x: 1 if x else 0)
y_train=hehe['lab'].as_matrix()
X_train, Y_train = shuffle(x_train, y_train)

main_input = Input(shape=(maxlen,))
embedder = Embedding(max_features,embedding_dims,input_length=maxlen)
embed=embedder(main_input)
cnn1=Convolution1D(256,3,padding='same',strides=1,activation='relu')(embed)
#model.add(GlobalMaxPooling1D())
cnn1=GlobalMaxPooling1D()(cnn1)
cnn2=Convolution1D(256,4,padding='same',strides=1,activation='relu')(embed)
cnn2=GlobalMaxPooling1D()(cnn2)
cnn3=Convolution1D(256,5,padding='same',strides=1,activation='relu')(embed)
cnn3=GlobalMaxPooling1D()(cnn3)
cnn=concatenate([cnn1,cnn2,cnn3],axis=-1)
flat=Dense(512)(cnn)
#model.add(Dense(250))
drop=Dropout(0.2)(flat)
main_output=Dense(1,activation='sigmoid')(drop)
model=Model(inputs=main_input,outputs=main_output)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, random_state=0)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train, y_train,batch_size=32,epochs=3,validation_data=(X_test, y_test))

#23,25,26,27,42
lawlist=[44,45,47,52,53,61,62,64,65,67,68,69,70,72,73,77,78,79,133,141,196,224,225,263,264,266,267,274,303,312,345,347,348,354,356,383]
for law in lawlist:
	print(law)
	pos_row = set(label[label['label']==law]['row_id'])
	hehe=pd.DataFrame()
	hehe['lab'] = trainrow['row_id'].isin(pos_row)
	hehe['lab'] = hehe['lab'].apply(lambda x: 1 if x else 0)
	y_train=hehe['lab'].as_matrix()
	X_train, Y_train = shuffle(x_train, y_train)
	model = Sequential()
	model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
	model.add(Dropout(0.15))
	model.add(Conv1D(260,4,padding='valid',activation='relu',strides=1))
	model.add(GlobalMaxPooling1D())
	model.add(Dense(260))
	model.add(Dropout(0.2))
	model.add(Activation('relu'))
	# We project onto a single unit output layer, and squash it with a sigmoid:
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
	model.add(Conv1D(260,4,padding='valid',activation='relu',strides=1))
	model.add(GlobalMaxPooling1D())
	for i in range(len(model.layers)):
		model.layers[i].set_weights(weights[i])

	trainfeature = model.predict(x_train)
	testfeature = model.predict(x_test)
	np.save('/home/deeplearning2/ai/feature/trainf'+str(law),trainfeature)
	np.save('/home/deeplearning2/ai/feature/testf'+str(law),testfeature)
	K.clear_session()
	
