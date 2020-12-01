# 신경망학습 : 신경망에서의 기울기


import numpy as np
import os
import sys
from pathlib import Path

try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import softmax, sigmoid, cross_entropy_error

except ImportError:
    print('Library Module Can Not Found')


x = np.array([0.6, 0.9])        # 입력(x)              2 vector
t = np.array([0., 0., 1.])      # label(one-hot)      3 vector

# params = {
#     'w1' : np.array([[0.02, 0.224, 0.135], [0.01, 0.052, 0.345]]),
#     'b1' :np.array([0.45, 0.23, 0.11])
# }

params = dict()

def initialize(sz_input, sz_hidden, sz_output, w_init=0.01):
    params['w1'] = w_init * np.random.randn(sz_input, sz_hidden)
    params['b1'] = np.zeros(sz_hidden)
    params['w2'] = w_init * np.random.randn(sz_hidden, sz_output)
    params['b2'] = np.zeros(sz_output)


def forward_propagation(x):
    w1 = params['w1']
    b1 = params['b1']
    a1 = np.dot(x, w1) + b1

    z1 = sigmoid(a1)

    w2 = params['w2']
    b2 = params['b2']
    a2 = np.dot(z1, w2) + b2

    y = softmax(a2)

    return y


def loss(x, t):                        #softmax
    y = forward_propagation(x)
    e = cross_entropy_error(y, t)

    return e

def accuracy(x, t):
    y = forward_propagation(x)
    y = np.argmax(y, axis = 1)
    t = np.argmax(t, axis = 1)

    acc = np.sum(y == t) / float(x.shape[0])

    return acc

def numerical_gradient_net(x, t):
    h = 1e-4
    gradient = dict()

    for key in params:
        param = params[key]
        param_gradient = np.zeros_like(param)

        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp = param[idx]

            param[idx] = tmp + h
            h1 = loss(x, t)

            param[idx] = tmp - h
            h2 = loss(x, t)

            param_gradient[idx] = (h1 - h2) / (2 * h)

            param[idx] = tmp

            it.iternext()
        gradient[key] = param_gradient

    return gradient


