import mnist
import numpy as np
import math
import random
import pickle
from matplotlib import pyplot as plt

data = mnist.MNIST('./data/mnist')

train_X, train_y = data.load_training()
train_X = np.array(train_X) / 255.0
train_y = np.eye(np.max(train_y) + 1)[train_y]

test_X, test_y = data.load_testing()
test_X = np.array(test_X) / 255.0
test_y = np.array(test_y)

def tanh(x):
    return np.tanh(x)

def tanh_prime(y):
    return 1 - y**2

def sigmoid_prime(y):
    return y * (1 - y)

class NeuralNetwork:
    def __init__(self, train_X, train_y, test_X, test_y, learning_rate=0.01, alpha=2):
        self.train_X = train_X
        self.train_y = train_y

        self.test_X = test_X
        self.test_y = test_y

        num_outputs = train_y.shape[1]
        num_inputs = train_X.shape[1]
        self.num_samples = len(train_X)

        layer_size = int(self.num_samples / (alpha * (num_inputs+num_outputs)))

        self.wh = np.random.randn(layer_size, num_inputs) * np.sqrt(1/num_inputs)
        self.bh = np.random.randn(layer_size) * np.sqrt(1/num_inputs)
        self.lh_a = []

        self.w_out = np.random.randn(num_outputs, layer_size) * np.sqrt(1/layer_size)
        self.b_out = np.random.randn(num_outputs) * np.sqrt(1/layer_size)
        self.out = []

        self.learning_rate = learning_rate

        self.cost = []
    
    def forward(self, a):
        self.lh_a = tanh(self.wh @ a + self.bh)
        
        self.out = tanh(self.w_out @ self.lh_a + self.b_out)

    def backward(self, a, yi):
        out_error = self.out - yi
        lh_error = np.dot(self.w_out.T, out_error)

        self.cost.append(np.mean(out_error**2))

        out_delta = self.learning_rate * out_error * tanh_prime(self.out)
        out_adj = np.dot(out_delta.reshape(-1, 1), self.lh_a.reshape(-1, 1).T)

        lh_delta = self.learning_rate * lh_error * tanh_prime(self.lh_a)
        lh_adj = np.dot(lh_delta.reshape(-1, 1), a.reshape(-1, 1).T)

        self.w_out -= out_adj
        self.wh -= lh_adj

        self.b_out -= out_delta
        self.bh -= lh_delta

    def train(self, num_epochs=10):
        for i in range(num_epochs):
            for a, yi in zip(self.train_X, self.train_y):
                self.forward(a)
                self.backward(a, yi)

            print('Epoch: ', i+1)
            print('Cost: ', np.mean(self.cost))
            self.cost.clear()
    
    def test(self):
        num_correct = 0
        for a, yi in zip(self.test_X, self.test_y):
            nn.forward(a)

            plt.imshow(a.reshape(28, 28), cmap='gray')
            plt.show()

            ans = np.argmax(self.out)
            if ans == yi:
                num_correct += 1

        print('percent correct: ', num_correct / len(test_y) * 100)

#with open('./data/nn.pickle', 'rb') as f:
#    nn = pickle.load(f)
nn = NeuralNetwork(train_X, train_y, test_X, test_y)
nn.train()
nn.test()
