import cvxopt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import style
import math

style.use('ggplot')

class SVM:
    def __init__(self, data_dict):
        self.data_dict = data_dict

        # Stuff for visualization
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.colors = {
            1: 'r',
            -1: 'b',
        }

        # Extract data
        X = []
        y = []

        self.all_data = []

        for yi in data_dict:
            for xi in data_dict[yi]:
                X.append(xi)
                y.append(yi)

                for feature in xi:
                    self.all_data.append(feature)
        
        self.X = np.array(X) * 1.
        self.y = np.array(y).reshape(-1, 1) * 1.

        m,_ = self.X.shape 
        X_dash = self.y * self.X 

        H = X_dash @ X_dash.T 

        P = cvxopt.matrix(H)
        q = cvxopt.matrix(-np.ones((m, 1)))
        G = cvxopt.matrix(-np.eye(m))
        h = cvxopt.matrix(np.zeros(m))
        A = cvxopt.matrix(self.y.reshape((1, -1)))
        b = cvxopt.matrix(np.zeros(1))

        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.lagrange_multipliers = np.array(sol['x'])
        self.w = ((self.y.reshape(-1, 1) * self.lagrange_multipliers).T @ self.X).flatten()

        sv = (self.lagrange_multipliers > 1e-4).flatten()

        self.b = np.mean(self.y[sv] - np.dot(self.X[sv], self.w.reshape(-1, 1)))

        print('lagrange multipliers: ', self.lagrange_multipliers[sv])
        print('w: ', self.w)
        print('b: ', self.b)

    def predict(self, u):
        classification =  np.sign(np.dot(self.w, u) + self.b)

        self.ax.scatter(u[0], u[1], s=100, marker='*', c=self.colors[classification])

        return classification
    
    def visualize(self):
        for yi in self.data_dict:
            for xi in self.data_dict[yi]:
                self.ax.scatter(xi[0], xi[1], s=100, color=self.colors[yi])
        
        def hyperplane(x, w, b, v): 
            return (v - b - w[0]*x) / w[1] 
        data_range = (0.9 * np.amin(self.all_data), 1.1 * np.amax(self.all_data))
        hyp_x_min = data_range[0]
        hyp_x_max = data_range[1]

        # Positive sv hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)

        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2])
        
        # Negative sv hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1) 
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)

        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2])

        # Decision boundary
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, 0)

        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2])

        plt.show()

data_dict = {
    1: [
        [1, 5],
        [2, 6],
        [4, 9]
    ],

    -1: [
        [3, 15],
        [4, 20]
    ]
}

svm = SVM(data_dict)

svm.predict([3, 8])
svm.predict([3, 4.5])

svm.visualize()
