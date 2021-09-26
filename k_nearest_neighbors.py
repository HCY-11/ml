import numpy as np
import pandas as pd
import math

class KNN:
    def __init__(self, k, new_point):
        self.k = k
        self.new_point = new_point
        self.distances = []
    
    def train(self, X_train, y_train):
        for i in range(len(X_train)):
            dist = math.sqrt(np.sum((X_train[i] - self.new_point)**2))

            self.distances.append((dist, i))
        
        self.distances = sorted(self.distances)[:self.k]

    def predict(self):
        malignant = 0
        benign = 0
        for _, i in self.distances:
            if y_train[i] == 4:
                malignant += 1
            else:
                benign += 1

        return 2 if benign > malignant else 4
        

if __name__ == "__main__":
    df = pd.read_csv('data/breast-cancer-wisconsin.csv')
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.drop('id', 1)

    y_train = np.array(df['class'], dtype=float)
    X_train = np.array(df.drop('class', 1), dtype=float)

    new_point = [ 10, 6, 8, 7, 1, 4, 6, 7, 1 ]
    knn = KNN(13, new_point)
    knn.train(X_train, y_train)
    print(knn.predict())