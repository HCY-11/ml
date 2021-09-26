import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x)**2

def cross_entropy(y_hat_i):
    return -np.log(y_hat_i)

class Parameter:
    def __init__(self, name, value):
        self.name = name
        self.v = value
        self.d = np.zeros_like(value)
        self.m = np.zeros_like(value)

class LSTM_RNN: 
    def __init__(self, num_features, layer_size, learning_rate=0.001, t_stepsize=25):
        self.learning_rate = learning_rate
        self.num_features = num_features
        self.layer_size = layer_size
        self.t_stepsize = t_stepsize;

        self.h = np.zeros((layer_size, 1))
        self.C = np.zeros((layer_size, 1))

        self.W_f = Parameter('W_f', np.random.randn(layer_size, num_features + layer_size) / np.sqrt(layer_size))
        self.b_f = Parameter('b_f', np.random.randn(layer_size, 1) / np.sqrt(layer_size))

        self.W_i = Parameter('W_i', np.random.randn(layer_size, num_features + layer_size) / np.sqrt(layer_size))
        self.b_i = Parameter('b_i', np.random.randn(layer_size, 1) / np.sqrt(layer_size))

        self.W_c = Parameter('W_c', np.random.randn(layer_size, num_features + layer_size) / np.sqrt(layer_size))
        self.b_c = Parameter('b_c', np.random.randn(layer_size, 1) / np.sqrt(layer_size))

        self.W_o = Parameter('W_o', np.random.randn(layer_size, num_features + layer_size) / np.sqrt(layer_size))
        self.b_o = Parameter('b_o', np.random.randn(layer_size, 1) / np.sqrt(layer_size))

        self.W_y = Parameter('W_y', np.random.randn(num_features, layer_size) / np.sqrt(num_features))
        self.b_y = Parameter('b_y', np.random.randn(num_features, 1) / np.sqrt(num_features))
    
    def forward(self, x, h_prev, C_prev):
        z = np.row_stack((x, self.h))

        f = sigmoid(np.dot(self.W_f.v, z) + self.b_f.v)

        i = sigmoid(np.dot(self.W_i.v, z) + self.b_i.v)

        c_bar = tanh(np.dot(self.W_c.v, z) + self.b_c.v)

        C = f * C_prev + i * c_bar

        o = sigmoid(np.dot(self.W_o.v, z) + self.b_o.v)

        h = o * tanh(C)

        v = np.dot(self.W_y.v, h) + self.b_y.v
        y = np.exp(v) / np.sum(np.exp(v))

        return z, f, i, c_bar, C, o, h, v, y
    
    def backward(self, target, dh_next, dC_next, C_prev,
             z, f, i, c_bar, C, o, h, v, y,):
        dv = np.copy(y)
        dv[target] -= 1

        self.W_y.d += np.dot(dv, self.h.T)
        self.b_y.d += dv

        dh = np.dot(self.W_y.v.T, dv)
        dh += dh_next

        do = dh * tanh(C) * sigmoid_prime(o)

        self.W_o.d += np.dot(do, z.T)
        self.b_o.d += do

        dC = dh * o * tanh_prime(tanh(C))
        dC += dC_next
        dc_bar = dC * i * tanh_prime(c_bar)

        self.W_c.d += np.dot(dC, z.T)
        self.b_c.d += dC

        di = dC * c_bar * sigmoid_prime(i)

        self.W_i.d += np.dot(di, z.T)
        self.b_i.d += di

        df = dC * C_prev * sigmoid_prime(f)

        self.W_f.d += np.dot(df, z.T)
        self.b_f.d += df

        dz = np.dot(self.W_f.v.T, df) + np.dot(self.W_i.v.T, di) + np.dot(self.W_c.v.T, dc_bar) + np.dot(self.W_o.v.T, do)
    
        dh_prev = dz[:self.layer_size, :]
        dC_prev = f * dC

        return dh_prev, dC_prev

    def clip_gradients(self):
        for p in [ self.W_o, self.b_o, self.W_i, self.b_i, self.W_y, self.b_y, self.W_f, self.b_f, self.W_c, self.b_c ]:
            np.clip(p.d, -1, 1, out=p.d)

    def update_parameters(self):
        for p in [ self.W_o, self.b_o, self.W_i, self.b_i, self.W_y, self.b_y, self.W_f, self.b_f, self.W_c, self.b_c ]:
            p.m += p.d**2
            p.v -= self.learning_rate * p.d / np.sqrt(p.m + 1e-8)

    def clear_gradients(self):
        for p in [ self.W_o, self.b_o, self.W_i, self.b_i, self.W_y, self.b_y, self.W_f, self.b_f, self.W_c, self.b_c ]:
            p.d.fill(0)

    def forward_backward(self, inputs, targets, h_prev, C_prev):
        x_s, z_s, f_s, i_s,  = {}, {}, {}, {}
        c_bar_s, C_s, o_s, h_s = {}, {}, {}, {}
        v_s, y_s =  {}, {}

        h_s[-1] = np.copy(h_prev)
        C_s[-1] = np.copy(C_prev)
        
        loss = 0

        with open('./output.txt', 'a') as f:
            for t in range(len(inputs)):
                x_s[t] = np.zeros((self.num_features, 1))
                x_s[t][inputs[t]] = 1 # Input character
                
                (z_s[t], f_s[t], i_s[t],
                c_bar_s[t], C_s[t], o_s[t], h_s[t],
                v_s[t], y_s[t]) = \
                    self.forward(x_s[t], h_s[t - 1], C_s[t - 1]) # Forward pass
                    
                loss += cross_entropy(y_s[t][targets[t], 0])

                out_char = idx_to_char[np.argmax(y_s[t])]
                f.write(str(out_char))
                
            self.clear_gradients()

            dh_next = np.zeros_like(h_s[0]) #dh from the next character
            dC_next = np.zeros_like(C_s[0]) #dh from the next character

        for t in reversed(range(len(inputs))):
            # Backward pass
            dh_next, dC_next = self.backward(target = targets[t], dh_next = dh_next,
                        dC_next = dC_next, C_prev = C_s[t-1],
                        z = z_s[t], f = f_s[t], i = i_s[t], c_bar = c_bar_s[t],
                        C = C_s[t], o = o_s[t], h = h_s[t], v = v_s[t],
                        y = y_s[t])

        self.clip_gradients()
            
        return loss, h_s[len(inputs) - 1], C_s[len(inputs) - 1]
    
    def train(self, train_X):
        pointer = 0
        iteration = 0

        for _ in range(int(10000)):
            # Reset
            if pointer + self.t_stepsize >= len(train_X) or iteration == 0:
                g_h_prev = np.zeros((self.layer_size, 1))
                g_C_prev = np.zeros((self.layer_size, 1))
                pointer = 0

            inputs = ([char_to_idx[ch] 
                       for ch in train_X[pointer: pointer + self.t_stepsize]])
            targets = ([char_to_idx[ch] 
                        for ch in train_X[pointer + 1: pointer + self.t_stepsize + 1]])

            loss, g_h_prev, g_C_prev = \
                self.forward_backward(inputs, targets, g_h_prev, g_C_prev)
            
            if iteration % 100 == 0:
                print('Loss: ', loss / len(inputs))
                print()

            self.update_parameters()

            pointer += self.t_stepsize
            iteration += 1

data = open('./data/lstm-data.txt', 'r').read()

chars = list(set(data))
num_samples, num_features = len(data), len(chars)

char_to_idx = { ch:i for i, ch in enumerate(chars) }
idx_to_char = { i:ch for i, ch in enumerate(chars) }

lstm = LSTM_RNN(num_features, 1000)

lstm.train(data)
