import mnist
import numpy as np
import math
import cnn_jit
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt

mnist = mnist.MNIST('./data/mnist/fashion')

train_X, train_y = mnist.load_training()
train_X = np.array(train_X).reshape((60000, 28, 28))
train_y = np.array(train_y)

test_X, test_y = mnist.load_testing()
test_X = np.array(test_X).reshape((10000, 28, 28))
test_y = np.array(test_y)

class CNN:
    def __init__(self, train_X, train_y, test_X, test_y, 
                learning_rate=0.01,
                num_filters=7,
                filter_size=5,
                batch_size=4,
                decay_rate=0.9,
                drop_time=10):
        self.train_X = train_X / 255.0
        self.train_y = train_y
        self.test_X = test_X / 255.0
        self.test_y = test_y

        self.conv_w = np.random.randn(num_filters, filter_size, filter_size) * np.sqrt(2) / filter_size
        self.conv_b = np.random.randn(num_filters) * np.sqrt(2) / filter_size

        self.conv_w_adj = np.zeros_like(self.conv_w)
        self.conv_b_adj = np.zeros_like(self.conv_b)

        feature_map_size = train_X[0].shape[0] - filter_size + 1
        self.feature_maps = np.zeros((num_filters, feature_map_size, feature_map_size))

        pool_size = feature_map_size // 2
        self.max_pools = np.zeros((num_filters, pool_size, pool_size))
        self.max_indices = np.zeros((num_filters, pool_size, pool_size, 2), dtype=np.int)
        self.num_filters = num_filters

        fc_size = 10
        in_size = num_filters * pool_size**2

        self.fc_w = np.random.randn(fc_size, in_size) * np.sqrt(2) / pool_size
        self.fc_b = np.random.randn(fc_size) * np.sqrt(2) / pool_size

        self.fc_w_adj = np.zeros_like(self.fc_w)
        self.fc_b_adj = np.zeros_like(self.fc_b)

        self.out = []
        self.loss = []

        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.lr_decay = decay_rate
        self.drop_time = drop_time

    def __conv(self, Xi):
        self.feature_maps = cnn_jit.conv_jit(Xi, self.conv_w, self.conv_b)
    
    def __conv_back(self, max_pool_back, Xi):
        conv_adj, b_adj = cnn_jit.conv_back_jit(Xi, max_pool_back, self.feature_maps, self.conv_w.shape)

        self.conv_w_adj += conv_adj
        self.conv_b_adj += b_adj

    def __relu(self):
        self.feature_maps = cnn_jit.relu(self.feature_maps)

    def __max_pool(self):
        self.max_pools, self.max_indices = cnn_jit.max_pool_jit(self.feature_maps, self.num_filters)

    def __max_pool_back(self, fc_err):
        out = cnn_jit.max_pool_back_jit(fc_err, self.fc_w, self.max_indices, self.feature_maps.shape, self.num_filters)

        return out

    def __fully_connected(self):
        self.out = cnn_jit.fully_connected_jit(self.max_pools, self.fc_w, self.fc_b)
    
    def __fully_connected_back(self, yi):
        w_adj, b_adj, loss, err = cnn_jit.fully_connected_back_jit(self.out, yi, self.max_pools)

        self.fc_w_adj += w_adj
        self.fc_b_adj +=  b_adj

        self.loss.append(loss)

        return err

    def forward(self, Xi):
        self.__conv(Xi)
        self.__relu()
        self.__max_pool()
        self.__fully_connected()

    def backward(self, Xi, yi):
        fc_error = self.__fully_connected_back(yi)
        max_feature_map = self.__max_pool_back(fc_error)
        self.__conv_back(max_feature_map, Xi)
    
    def update(self, lr):
        self.fc_w -= lr * self.fc_w_adj
        self.fc_b -= lr * self.fc_b_adj

        self.fc_w_adj = np.zeros_like(self.fc_w)
        self.fc_b_adj = np.zeros_like(self.fc_b)

        self.conv_w -= lr * self.conv_w_adj
        self.conv_b -= lr * self.conv_b_adj

        self.conv_w_adj = np.zeros_like(self.conv_w)
        self.conv_b_adj = np.zeros_like(self.conv_b)

    def train(self, num_epochs=2):
        lr = self.learning_rate
        num_samples = len(self.train_X)

        for i in range(num_epochs):
            epoch = i + 1
            print('\nEpoch: ', epoch)
            for j in tqdm(range(num_samples), total=num_samples):
                self.forward(self.train_X[j])
                self.backward(self.train_X[j], self.train_y[j])

                batch_idx = j + 1
                if batch_idx % self.batch_size == 0:
                    self.update(lr)
            
            print('Loss: ', np.mean(self.loss))
            self.loss.clear()

            if epoch % self.drop_time == 0:
                lr *= self.lr_decay

    def test(self):
        num_correct = 0
        print('\n\nTEST RESULTS: ')

        num_samples = len(self.test_X)
        
        for i in tqdm(range(num_samples), total=num_samples):
            self.forward(self.test_X[i])

            guess = np.argmax(self.out)

            ans = test_y[i]

            if guess == ans:
                num_correct += 1
        
        print('(% correct): ', num_correct / num_samples * 100)

cnn = CNN(train_X, train_y, test_X, test_y)
cnn.train()

row = 3
col = 4
_, ax = plt.subplots(row, col)

k = 0
for i in range(row):
    for j in range(col):
        if k < 7:
            ax[i][j].imshow(cnn.feature_maps[k], cmap='gray')
            k += 1
        else:
            ax[i][j].imshow(train_X[-1], cmap='gray')

plt.show()

cnn.test()

with open('./cnn.pickle', 'wb') as f:
    pickle.dump(CNN, f)