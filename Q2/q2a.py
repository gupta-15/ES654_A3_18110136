from Logistic_Regression import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_classification
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
X, y = make_classification(n_samples=1000)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
clf = LogisticRegression().l1_fit(X_train, y_train)
y_hat = (clf.predict(X_test))
ans = clf.score1(y_hat,y_test)
print("For L1 regularised Logistic Regression ")
print(ans)

clf = LogisticRegression().l2_fit(X_train, y_train)
y_hat = (clf.predict(X_test))
ans = clf.score2(y_hat,y_test)
print("For L2 regularised Logistic Regression ")
print(ans)
