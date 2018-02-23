import pandas as pd
import numpy as np

#trainf = pd.read_csv("../bagwords/trainfeature_2.csv")
#testf = pd.read_csv("../bagwords/testfeature_2.csv")
xtrain = np.load('/home/deeplearning2/ai/code/newf1.npy')
a=np.load('/home/deeplearning2/ai/code/trainlaw.npy')
xtrain=np.concatenate([xtrain,a],axis=1)
xtest = np.load('/home/deeplearning2/ai/code/newf2.npy')
b=np.load('/home/deeplearning2/ai/code/testlaw.npy')
xtest=np.concatenate([xtest,b],axis=1)
label = pd.read_csv("/home/deeplearning2/ai/code/label.csv")
ytrain=label['label'].as_matrix()
#ans=[]
#xtrain = trainf.drop(['row_id','money'],axis=1).as_matrix()
#ytrain = trainf['money'].as_matrix()
#xtest=testf.drop('row_id',axis=1).as_matrix()
#print(xtest)
import xgboost as xgb
params = {
            'objective': 'multi:softmax',
            'eta': 0.05,
            'max_depth': 15,
            'eval_metric': 'merror',
            'num_class':9,
            'missing': 0,
            'silent' : 1,
            'nthread':10
            }
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(xtrain, ytrain, random_state=0)
xgbtrain = xgb.DMatrix(xtrain, ytrain)
xgbtest = xgb.DMatrix(xtest)
watchlist = [ (xgbtrain,'train'), (xgbtrain, 'test') ]
num_rounds=300
model = xgb.train(params, xgbtrain, num_rounds)

ans=[]
probas=model.predict(xgbtest)
for i in range(len(probas)):
	ans.append(probas[i])

ansD = pd.DataFrame(ans,columns=['money'],index=None)
ansD.to_csv("../money-12-2.csv",index=None)
'''
