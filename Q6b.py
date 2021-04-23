import numpy as np
import matplotlib.pyplot as plt
from Q5 import NN_Regression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold
import pandas as pd

data = load_boston()                                                            
X,Y = data["data"], data["target"]    
kf = KFold(n_splits=3)                                           
for train_index,test_index in kf.split(X):
    X_train,X_test = X[train_index], X[test_index]
    y_train,y_test = Y[train_index], Y[test_index]

layers = [13,8,6,1]                                                      
num_iters = 1000                                                             
learning_rate = 0.05                                                             
params = NN_Regression().model(X_train, y_train, layers, num_iters, learning_rate)           
train_acc, test_acc = NN_Regression().score(X_train, X_test, y_train, y_test, params,layers)  
print('RMSE Training Data = ' + str(train_acc))
print('RMSE Test Data = ' + str(test_acc))