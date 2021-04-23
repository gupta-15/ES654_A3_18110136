from Logistic_Regression import LogisticRegression
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
N = 50
P = 8
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(0,2,N))
clf = LogisticRegression().fit(X, y)
y_hat = (clf.predict(X))
ans = clf.score1(y_hat,y)
print("Accuracy with Gradient_Descent Normally")
print(ans)

clf = LogisticRegression().fit_autograd(X, y)
y_hat = (clf.predict(X))
ans = clf.score2(y_hat,y)
print("Accuracy with Autograd Implementation")
print(ans)