import autograd.numpy as np
from autograd import grad
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as cma

### REfrences :- https://towardsdatascience.com/logistic-regression-from-scratch-69db4f587e17

class LogisticRegression(object):

    def __init__(self, learning_rate=0.05, max_iter=20,l1_coef = 0.5,l2_coef = 0.5):
        self.learning_rate  = learning_rate
        self.max_iter       = max_iter
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
    
    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1] + 1)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        for _ in range(self.max_iter):
            y_hat = self.sigmoid(np.dot(X,self.theta))
            errors = y_hat - y
            N = X.shape[1]

            delta_grad = self.learning_rate * (np.dot(X.T,errors))
            self.theta -= delta_grad / N
                
        return self
    
    def fit_autograd(self, X, y):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        X=np.array(X).astype(np.float64)
        y=np.array(y).reshape(-1,1).astype(np.float64)
        self.theta=np.zeros(X.shape[1]).reshape(-1,1)
        def loss(theta,X,y):
            y_hat = self.sigmoid(np.dot(X,theta))
            y_hat = np.squeeze(y_hat)
            y = np.squeeze(y)
            res = -np.sum(y.dot(np.log10(y_hat))+(1-y).dot(np.log10(1-y_hat)))
            res = res/X.shape[0]
            return res
        change = grad(loss)

        for _ in range(self.max_iter):
            gradient = change(self.theta,X,y)
            self.theta -= self.learning_rate*gradient
            #print(theta)

        return self

    def l2_fit(self, X, y):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        X=np.array(X).astype(np.float64)
        y=np.array(y).reshape(-1,1).astype(np.float64)
        self.theta=np.zeros(X.shape[1]).reshape(-1,1)
        def loss(theta,X,y):
            y_hat = self.sigmoid(np.dot(X,theta))
            y_hat = np.squeeze(y_hat)
            y = np.squeeze(y)
            error = -np.sum(y.dot(np.log10(y_hat))+(1-y).dot(np.log10(1-y_hat)))
            error = error/X.shape[0]
            error+= self.l2_coef/(2*X.shape[0])*np.sum(np.square(theta))
            return error
        change = grad(loss)

        for _ in range(self.max_iter):
            gradient = change(self.theta,X,y)
            self.theta -= self.learning_rate*gradient
        return self
    
    def l1_fit(self, X, y):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        X=np.array(X).astype(np.float64)
        y=np.array(y).reshape(-1,1).astype(np.float64)
        self.theta=np.zeros(X.shape[1]).reshape(-1,1)
        def loss(theta,X,y):
            y_hat = self.sigmoid(np.dot(X,theta))
            y_hat = np.squeeze(y_hat)
            y = np.squeeze(y)
            error = -np.sum(y.dot(np.log10(y_hat))+(1-y).dot(np.log10(1-y_hat)))
            error = error/X.shape[0]
            error+= self.l1_coef/(2*X.shape[0])*np.sum(np.abs(theta))
            return error
        change = grad(loss)

        for _ in range(self.max_iter):
            gradient = change(self.theta,X,y)
            self.theta -= self.learning_rate*gradient
        return self

    def predict_proba(self, X):
        temp =  self.theta[0] + np.dot(X,self.theta[1:])
        #print(temp)
        return self.sigmoid(temp)
    
    def predict(self, X):
        return np.round(self.predict_proba(X))
        
    def sigmoid(self, z):
        return 1/(1+ np.exp(-z))

    def get_params(self):
        try:
            params = dict()
            params['intercept'] = self.theta[0]
            params['coef'] = self.theta[1:]
            return params
        except:
            raise Exception('Fit the model first!')

    def plot_decision_boundary(self, X, y):
        cMap = cma.ListedColormap(["#6b76e8", "#c775d1"])
        cMapa = cma.ListedColormap(["#c775d1", "#6b76e8"])
        print("DOne")
        temp_X = pd.DataFrame(X)
        temp_y = pd.Series(y)
        X_cord = temp_X.columns[0]
        y_cord = temp_X.columns[1]
        x_min = temp_X[X_cord].min() - .5
        x_max = temp_X[X_cord].max() + .5
        y_min = temp_X[y_cord].min() - .5
        y_max = temp_X[y_cord].max() + .5
        h = 0.02
        print("Done 2")
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        clf = LogisticRegression().fit(X, y)
        Z = clf.predict(np.column_stack((xx.ravel(), yy.ravel())))
        Z = Z.reshape(xx.shape)
        plt.figure(1, figsize=(8, 6), frameon=True)
        plt.axis('off')
        plt.pcolormesh(xx, yy, Z, cmap=cMap)
        plt.scatter(temp_X[X_cord], temp_X[y_cord], c = temp_y, cmap=cMapa, marker = "o",edgecolors='k', alpha=0.6)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.show()
    
    def score1(self,y_hat,y):
        answer = 0
        for i in range(len(y)):
            if int(y_hat[i]) == (y[i]):
                answer+=1
        return (answer/len(y_hat))
    
    def score2(self,y_hat,y):
        answer = 0
        for i in range(len(y)):
            if int(y_hat[i][0]) == (y[i]):
                answer+=1
        return (answer/len(y_hat))

    def plot_important_features(self):
        theta = (self.theta)
        x=[]
        y = []
        for i in range(len(theta)):
            x.append('θ'+str(i))
            y.append(abs(theta[i][0]))
        
        plt.bar(x,y)
        plt.title("Magnitude of theta for important features")
        plt.show()