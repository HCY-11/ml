import pandas as pd
import numpy as np
import random

class MultiVarLinearRegression:
    def __init__(self, train_X, train_y, num_features, learning_rate=0.005, n_iter=10000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

        self.bias = np.ones((len(train_X), 1))
        self.y = train_y # No need to normalize targets
        self.X = np.append(self.bias, self.normalize_features(train_X), 1) # Normalize and add bias column

        self.weights = np.array([ random.random() for _ in range(num_features + 1) ])
    
    def normalize_features(self, X): 
        for feature in X.T:
            f_mean = np.mean(feature)
            f_range = np.amax(feature) - np.amin(feature)

            feature -= f_mean

            feature /= f_range

        return X

    def cost_func(self):
        total = (self.predict() - self.y)**2

        return total.sum() / (2*len(total))
    
    def predict(self):
        return np.dot(self.X, self.weights)
    
    def predict_new(self, X):
        self.X = X.T
        return self.predict()
    
    def update_weights(self):
        predictions = self.predict()
        errors = predictions - self.y

        gradient = np.dot(self.X.T, errors)

        gradient /= len(self.X)

        gradient *= self.learning_rate

        self.weights -= gradient
    
    def train(self):
        for _ in range(self.n_iter):
            self.update_weights()

df = pd.read_csv('data/data-large.csv')

df_y = np.array(df.get('sales'))
df_X = np.array(df.drop([ 'index', 'sales' ], 1))

lr = MultiVarLinearRegression(df_X, df_y, 3, learning_rate=0.01)
print(lr.cost_func())
lr.train()
print(lr.cost_func())