import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
import pickle
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#def init_network():
#    network = {}
#    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
#    network['b1'] = np.array([0.1, 0.2, 0.3])
#    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
#    network['b2'] = np.array([0.1, 0.2])
#    network['W3'] = np.array([[0.1, 0.3],[0.2, 0.4]])
#    network['b3'] = np.array([0.1, 0.2])

#    return network

#def forward(network, x):
#    W1, W2, W3 = network['W1'], network['W2'], network['W3']
#    b1, b2, b3 = network['b1'], network['b2'], network['b3']
#
#    a1 = np.dot(x, W1) + b1
#    z1 = sigmoid(a1)
#    a2 = np.dot(z1, W2) + b2
#    z2 = sigmoid(a2)
#    a3 = np.dot(z2, W3) + b3
#    y = identity_function(a3)

#    return y

#def identity_function(x1):
#    return x1

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # to prevent overflow
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

#network = init_network()
#x = np.array([1.0, 0.5])
#y = forward(network, x)
# print(y)

#(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

#print(x_train.shape)
#print(t_train.shape)
#print(x_test.shape)
#print(t_test.shape)

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("../ch03/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

# 2乗和誤差計算function
# yとtはnumpyの配列

def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)

# 交差エントロピー誤差計算function

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


#x, t = get_data()
#network = init_network()

#accuracy_cnt = 0
#for i in range(len(x)):
#    y = predict(network, x[i])
#    p = np.argmax(y)

#    if p == t[i]:
#        accuracy_cnt += 1

#print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

# batchで処理する場合
x, t = get_data()
network = init_network()

batch_size = 100 # バッチの数
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

# ミニバッチ用データの取得

#train_size = x_train.shape[0]
#batch_size = 10
#batch_mask = np.random.choice(train_size, batch_size)
#x_batch = x_train[batch_mask]
#t_batch = t_train[batch_mask]


