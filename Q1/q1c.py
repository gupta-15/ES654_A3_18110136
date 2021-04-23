from Logistic_Regression import LogisticRegression
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np

data = np.array(load_breast_cancer().data)
y = np.array(load_breast_cancer().target)
kf = KFold(n_splits=3)
for train_index,test_index in kf.split(data):
    X_train,X_test = data[train_index], data[test_index]
    y_train,y_test = y[train_index], y[test_index]

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.Series(y_train)
y_test = pd.Series(y_test)
clf = LogisticRegression().fit(X_train, y_train)
y_hat = list(clf.predict(X_test))
y_t = list(y_test)
print("Overall Accuracy with K = 3 Folds")
print(clf.score1(y_hat,y_t))
