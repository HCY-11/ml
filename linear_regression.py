import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random

class LinearRegression:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.weight = random.random()
        self.bias = random.random()
    
    # Half MSE
    def cost_func(self, X, y):
        total = 0

        for i in range(len(X)):
            total += (self.predict(X[i]) - y[i])**2
        
        return total / (2 * len(X))

    def w_deriv(self, xi, yi):
        # -2x(yi - (mxi+b))
        return -xi * (yi - xi*self.weight - self.bias)
    
    def b_deriv(self, xi, yi):
        # -2(yi - (mxi + b))
        return -(yi - xi*self.weight - self.bias)
    
    def predict(self, xi):
        return xi*self.weight + self.bias
    
    def update_weight(self, X, y):
        w_deriv = 0
        b_deriv = 0

        for i in range(len(X)):
            w_deriv += self.w_deriv(X[i], y[i])
            b_deriv += self.b_deriv(X[i], y[i])
        
        self.weight -= (w_deriv / len(X)) * self.learning_rate
        self.bias -= (b_deriv / len(X)) * self.learning_rate
    
    def train(self, n_iterations, X, y):
        cost = 0

        for i in range(n_iterations):
            self.update_weight(X, y)

df = pd.read_csv('data/data.csv')
X = np.array(df.drop(['index', 'sales'], 1))

lr = LinearRegression(0.001)

print(lr.weight)
print(lr.bias)
