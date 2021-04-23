import numpy as np
from sklearn.datasets import load_digits
from Q5 import NN_Classification
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston


sc = StandardScaler()
neurons= [64, 50 ,15, 10]
g = ['relu','relu','sigmoid']
digits= load_digits()
n_samples=len(digits.images)
X=digits.images.reshape((n_samples,-1))
y=digits.target



kf = KFold(n_splits=3)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_train= np.transpose(X_train)
    X_test= np.transpose(X_test)
    y_train=y_train.reshape(y_train.shape[0],1)
    y_test=y_test.reshape(y_test.shape[0],1)
    y_train= np.transpose(y_train)
    y_test= np.transpose(y_test)

    Y_train_=np.zeros((10,y_train.shape[1]))
    for i in range(y_train.shape[1]):
        Y_train_[y_train[0,i],i]=1
    Y_test_=np.zeros((10,y_test.shape[1]))
    for i in range(y_test.shape[1]):
        Y_test_[y_test[0,i],i]=1

    parameters = NN_Classification().model(digits=X_train, target=Y_train_, neurons=neurons, num_iterations = 30000,g=g)
    predictions_train_L = NN_Classification().predict(X_train, param=parameters,g=g)
    print("Accuracy on Train Dataset: "+ str(np.sum(predictions_train_L==y_train)/y_train.shape[1] * 100)+" %")
    predictions_test_L=NN_Classification().predict(X_test,param=parameters,g=g)
    print("Accuracy on Test Dataset : "+ str(np.sum(predictions_test_L==y_test)/y_test.shape[1] * 100)+" %")
