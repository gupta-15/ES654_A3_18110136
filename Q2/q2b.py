from Logistic_Regression import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_classification
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=100, random_state=np.random.RandomState(0))
X= pd.DataFrame(X)
y = pd.Series(y)
result=[]
x = []
for j in range(30):
    ans = []
    for i in range(5):
        X1 = X[0:i*20]
        X2 = X[(i+1)*20:]
        X_train = X1.append(X2)
        X_test = X[i*20:(i+1)*20]
        y1 = y[0:i*20]
        y2 = y[(i+1)*20:]
        y_train = y1.append(y2)
        y_test = y[i*20:(i+1)*20]
        clf = LogisticRegression(l1_coef= j*2).l1_fit(X_train,y_train)
        y_hat = clf.predict(X_test)
        y_t = list(y_test)
        answer = 0
        for i in range(len(y_t)):
            if int(y_hat[i]) == y_t[i]:
                answer+=1
        ans.append((answer)/len(y_t))
    result.append((sum(ans)/len(ans)))
    x.append(j*5)
    print(ans)
print(result)
plt.plot(x,result)
plt.xlabel("Panelty Coefficient")
plt.ylabel("Accuracy")
plt.show()


