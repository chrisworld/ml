import numpy as np

# graph layout
n_in = 5
n_hidden = 5
n_out = 5

# hyperparameters
learning_rate = 0.1
momentum = 0.5

# samples
n_sample = 300

def relu(x):
    if x > 0:
        return x
    else:
        return 0

def train(params, x):
    # forward
    A = np.dot(x, params['Weight1']) + params['Bias1']

# layers init
V = np.random.normal(scale=1, size=(n_in, n_hidden))
W = np.random.normal(scale=1, size=(n_hidden, n_out))
bv = np.zeros(n_hidden)
bw = np.zeros(n_out)
params = {"Weight1": V, "Weight2": W, "Bias1": bv, "Bias2": bw}




# data
in_data = np.random.binomial(1, 0.5, (n_sample, n_in))
train(params, in_data)
