import pandas as pd
import numpy as np

# Refrences = https://towardsdatascience.com/multiclass-classification-algorithm-from-scratch-with-a-project-in-python-step-by-step-guide-485a83c79992

class multiLogisticRegression(object):
    def __init__(self,learning_rate = 0.05, max_iter = 100):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit(self,X,y):
        X = pd.concat([pd.Series(1, index=X.index), X], axis=1)
        y_temp = pd.DataFrame(np.zeros([X.shape[0], len(y.unique())]))
        for i in range(0, len(y.unique())):
            for j in range(0, len(y_temp)):
                if y[j] == y.unique()[i]:
                    y_temp.iloc[j, i] = 1
                else: 
                    y_temp.iloc[j, i] = 0
        self.theta = np.zeros([X.shape[1], y_temp.shape[1]])
        self.theta = self.gradient(X, y_temp, self.theta)
        return self

    def predict(self,X):
        X = pd.concat([pd.Series(1, index=X.index), X], axis=1)
        theta = pd.DataFrame(self.theta[0])
        output = []
        for i in range(0, 10):
            temp = self.sigmoid(theta.iloc[:,i], X)
            output.append(temp)
        output=pd.DataFrame(output)
        return output

    def score(self,X,y_hat,y):
        X = pd.concat([pd.Series(1, index=X.index), X], axis=1)
        y_temp = pd.DataFrame(np.zeros([X.shape[0], len(y.unique())]))
        for i in range(len(y.unique())):
            for j in range(len(y_temp)):
                if y[j] == y.unique()[i]:
                    y_temp.iloc[j, i] = 1
                else: 
                    y_temp.iloc[j, i] = 0
        accuracy = 0
        for col in range(10):
            for row in range(len(y_temp)):
                if y_temp.iloc[row, col] == 1:
                    if y_hat.iloc[col, row] >= 0.5:
                        accuracy += 1
        accuracy = accuracy/len(X)
        return accuracy

    def gradient(self,X, y, theta):
        m = len(X)
        for _ in range(self.max_iter):
            for i in range(10):
                theta = pd.DataFrame(theta)
                temp = self.sigmoid(theta.iloc[:,i], X)
                for j in range(theta.shape[0]):
                    theta.iloc[j, i] -= (self.learning_rate/m) * np.sum((temp-y.iloc[:, i])*X.iloc[:, j])
                theta = pd.DataFrame(theta)
        return theta, self.predict_probability
    
    def sigmoid(self,theta, X):
        return 1 / (1 + np.exp(-(np.dot(theta, X.T))))

    def predict_probability(self,X, y, theta):
        y_hat = self.sigmoid(X, theta)
        return -(1/len(X)) * np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))