import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

# Coursera Machine LearningCourse
# Refrence: https://towardsdatascience.com/an-introduction-to-neural-networks-with-implementation-from-scratch-using-python-da4b6a45c05b

class NN_Classification(object):

    def __init__(self):
        self.X=None
        self.y=None

    def back_sigmoid(self, dA, Z):
        sigm = 1/(1+np.exp(-Z))
        dZ = dA * sigm * (1-sigm)
        return dZ

    def back_relu(self, dA, Z):
        dZ = np.array(dA) 
        dZ[Z <= 0] = 0
        return dZ

    def back_identity(self, dA, Z):
        return np.array(dA)

    def parameter_set(self,neurons):
        np.random.seed(42)
        param = {}
        n = len(neurons)

        for i in range(n-1):
            param['W' + str(i+1)] = np.random.randn(neurons[i+1], neurons[i]) * 0.001 
            param['B' + str(i+1)] = np.zeros((neurons[i+1], 1)) + 0.001
            
        return param
    
    def forward(self,A, W, b):
        Z = np.dot(W, A) + b
        return Z, (A,W,b)

    def forward_gi(self,A, W, b, g_i):

        if g_i=="identity":
            Z, temp1 = self.forward(A, W, b)
            A, temp2 = self.identity(Z)

        elif g_i == "sigmoid":  
            Z, temp1 = self.forward(A, W, b)
            A, temp2 = self.sigmoid(Z)
            
        elif g_i == "relu":
            Z, temp1 = self.forward(A, W, b)
            A, temp2 = self.relu(Z)
   
        return A, (temp1,temp2)

    def forward_model(self,X, param,g):
        values = []
        L = len(param) // 2     
        for l in range(1, L+1):
            x = X
            tmp = self.forward_gi(x, param['W' + str(l)], param['B' + str(l)], g_i=g[l-1])
            X = tmp[0]
            value = tmp[1]
            values.append(value)      
        return X, values
    
    def cost(self,y_hat, y):
        cost = -1 / (y.shape[1]) * np.sum(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))
        cost = np.squeeze(cost) 
        return cost

    def backward(self,z, temp):
        A_temp=  temp[0]
        W = temp[1]
        m = A_temp.shape[1]
        w = np.dot(z, np.transpose(A_temp)) / m
        b = (np.sum(z, axis=1, keepdims=True)) / m
        a_temp= np.dot(np.transpose(W), z)
        return a_temp, w, b

    def backward_gi(self,A, temp, g_i):
        tmp1 = temp[0]
        tmp2 = temp[1]
        if g_i =="identity":
            Z=self.back_identity(A,tmp2)
        elif g_i == "relu":
            Z = self.back_relu(A, tmp2)
        elif g_i == "sigmoid":
            Z = self.back_sigmoid(A, tmp2)
        A_prev, W, b = self.backward(Z, tmp1)
        return A_prev, W, b
        
    def backward_model(self,y_hat, y, values,g):
        grads = {}
        tmp = - (np.divide(y, y_hat) - np.divide(1 - y, 1 - y_hat))
        val = values[len(values)-1]
        tmp = self.backward_gi(tmp, val, 'sigmoid')
        grads["a" + str(len(values)-1)] = tmp[0]
        grads["w" + str(len(values))] = tmp[1]
        grads["b" + str(len(values))] = tmp[2]
        
        for i in reversed(range(len(values)-1)):
            val = values[i]
            tmp = self.backward_gi(grads["a" + str(i + 1)], val, g_i = g[i])
            grads["a" + str(i)] = tmp[0]
            grads["w" + str(i+1)] = tmp[1]
            grads["b" + str(i+1)] = tmp[2]
        return grads
    
    def update_parameters(self,param, grads, learning_rate):
        length = len(param) // 2 
        
        for i in range(length):
            param["W" + str(i+1)] = param["W" + str(i+1)] - learning_rate * grads["w" + str(i+1)]
            param["B" + str(i+1)] = param["B" + str(i+1)] - learning_rate * grads["b" + str(i+1)]
        return param
    
    def model(self,digits, target, neurons, learning_rate = 0.01, num_iterations = 1, g =None):
        costs = []
        param = self.parameter_set(neurons)
        for _ in range(0, num_iterations):
            tmp = self.forward_model(digits, param,g)
            y_hat = tmp[0]
            value = tmp[1]
            cost = self.cost(y_hat, target)
            grads =  self.backward_model(y_hat, target, value,g)
            param = self.update_parameters(param, grads, learning_rate)
            costs.append(cost)
        '''
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        '''
        
        return param
    
    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z)),Z

    def relu(self, Z):
        return np.maximum(0,Z),Z

    def identity(self, Z):
        return Z,Z

    def predict(self,X, param,g):
        tmp = self.forward_model(X,param,g)
        y_hat = tmp[0]
        y_pred=np.argmax(y_hat,axis=0)
        return y_pred.reshape(1,y_pred.shape[0])

class NN_Regression(object):
    def __init__(self):
        self.parameters=None
        self.neurons=None
        self.X=None
        self.y=None
    
    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z))

    def relu(self, Z):
        return np.maximum(0,Z)

    def identity(self, Z):
        return Z

    def parameter_set(self,neurons):
        param = {}
        n = len(neurons)
        for i in range(n-1):
            param['W' + str(i+1)] = np.random.randn(neurons[i+1], neurons[i]) * 0.001 
            param['B' + str(i+1)] = np.zeros((neurons[i+1], 1)) + 0.001
        return param

    def forward_pass(self,X_train, params):
        values = {}
        L = len(params)//2
        for i in range(1, L+1):
            if i==1:
                values['Z' + str(i)] = np.dot(params['W' + str(i)], X_train) + params['B' + str(i)]
                values['A' + str(i)] = self.relu(values['Z' + str(i)])
            else:
                values['Z' + str(i)] = np.dot(params['W' + str(i)], values['A' + str(i-1)]) + params['B' + str(i)]
                if i==L:
                    values['A' + str(i)] = values['Z' + str(i)]
                else:
                    values['A' + str(i)] = self.relu(values['Z' + str(i)])
        return values

    def cost(self,values, Y_train):
        layers = len(values)//2
        Y_pred = values['A' + str(layers)]
        cost = 1/(2*len(Y_train)) * np.sum(np.square(Y_pred - Y_train))
        return cost

    def backward_pass(self,params, values, X_train, Y_train):
        layers = len(params)//2
        m = len(Y_train)
        grads = {}
        for i in range(layers,0,-1):
            if i==layers:
                z = 1/m * (values['A' + str(i)] - Y_train)
            else:
                a = np.dot(np.transpose(params['W' + str(i+1)]), z)
                z = np.multiply(a, np.where(values['A' + str(i)]>=0, 1, 0))
            if i==1:
                grads['W' + str(i)] = 1/m * np.dot(z,np.transpose(X_train))
                grads['B' + str(i)] = 1/m * np.sum(z, axis=1,keepdims=True)
            else:
                grads['W' + str(i)] = 1/m * np.dot(z,np.transpose(values['A' + str(i-1)]))
                grads['B' + str(i)] = 1/m * np.sum(z, axis=1,keepdims=True)
        return grads

    def update_params(self, param, grads, lr):
        new_param = {}
        for i in range(1,len(param)//2+1):
            new_param['W' + str(i)] = param['W' + str(i)] - lr * grads['W' + str(i)]
            new_param['B' + str(i)] = param['B' + str(i)] - lr * grads['B' + str(i)]
        return new_param

    def model(self,X_train, Y_train, layer_sizes, num_iters, lr):
        params = self.parameter_set(layer_sizes)
        for _ in range(num_iters):
            values = self.forward_pass(np.transpose(X_train), params)
            grads = self.backward_pass(params, values,np.transpose(X_train), np.transpose(Y_train))
            params = self.update_params(params, grads, lr)
        return params

    def score(self,X_train, X_test, y_train, y_test, params,layer_sizes):

        values_train = self.forward_pass(np.transpose(X_train), params)
        values_test = self.forward_pass(np.transpose(X_test), params)
        y_train_hat = values_train['A' + str(len(layer_sizes)-1)]
        y_test_hat = values_test['A' + str(len(layer_sizes)-1)]
        train_score = mse(y_train, np.transpose(y_train_hat))
        test_score = mse(y_test, np.transpose(y_test_hat))
        return train_score, test_score