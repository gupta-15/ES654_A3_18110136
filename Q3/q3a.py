from sklearn.datasets import load_digits
from kclass import multiLogisticRegression
import pandas as pd

df = pd.DataFrame(load_digits().data)
y = pd.Series(load_digits().target)

clf = multiLogisticRegression().fit(df,y)
y_hat = clf.predict(df)
ans = clf.score(df,y_hat,y)
print(ans)
