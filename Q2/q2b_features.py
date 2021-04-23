from Logistic_Regression import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_classification
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

rng = np.random.RandomState(0)
X, y = make_classification(n_samples=100, random_state=rng)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
clf = LogisticRegression(l1_coef = 5).l1_fit(X_train, y_train)
clf.plot_important_features()