from sklearn.datasets import load_digits
from kclass import multiLogisticRegression
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

X = np.array(load_digits().data)
y = np.array(load_digits().target)
skf = StratifiedKFold(n_splits =4)
for train_index, test_index in skf.split(X,y):
    X_train,X_test = X[train_index], X[test_index]
    y_train,y_test = y[train_index], y[test_index]
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.Series(y_train)
y_test = pd.Series(y_test)
clf = multiLogisticRegression().fit(X_train,y_train)
y_hat = clf.predict(X_test)
y_test = (list(y_test))
ans = []
for i in range(y_hat.shape[1]):
    m = max(list(y_hat[i]))
    m_index = (list(y_hat[i])).index(m)
    ans.append(m_index)
#print(ans)
c_matrix = []
for i in range(10):
    temp = []
    for j in range(10):
        temp.append(0)
    c_matrix.append(temp)
#print(type(ans[0]))
for i in range(len(ans)):
    c_matrix[int(ans[i])][y_test[i]] +=1
print(c_matrix)
accuracy = 0
for i in range(10):
    accuracy+= c_matrix[i][i]
print(accuracy/len(y_test))