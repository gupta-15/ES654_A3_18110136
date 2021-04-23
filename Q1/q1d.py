from Logistic_Regression import LogisticRegression
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.DataFrame(load_breast_cancer().data)
df = data.sample(n=2,axis='columns')
y = pd.Series(load_breast_cancer().target)
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.30, random_state=42)
LogisticRegression().plot_decision_boundary(X_train, y_train)


