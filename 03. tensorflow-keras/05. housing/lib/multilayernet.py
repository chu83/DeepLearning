# 신경망학습 : 신경망에서의 기울기


import numpy as np
import os
import sys
from pathlib import Path

try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from layers import ReLU, Affine, SoftmaxWithLoss
except ImportError:
    print('Library Module Can Not Found')


params = dict()
layers = []

def initialize(input_size, hidden_size, output_size, init_weight=0.01, init_params=None):
    hidden_count = len(hidden_size)


    if init_params is None:
        params['w1'] = init_weight * np.random.randn(input_size, hidden_size[0])
        params['b1'] = np.zeros(hidden_size[0])

        for idx in range(1, hidden_count):
            params[f'w{idx+1}'] = init_weight * np.random.randn(hidden_size[idx-1], output_size[idx])
            params[f'b{idx+1}'] = np.zeros(hidden_size[idx])

        params[f'w{hidden_count+1}'] = init_weight * np.random.randn(hidden_size[idx-1], output_size[idx])
        params[f'b{hidden_count+1}'] = np.zeros(output_size)
    else:
        globals()['params'] = init_params   #전역, 네임테이블(전역은 쓰지 않는게 좋다..)

    layers.append(Affine(params['w1'], params['b1']))
    layers.append(ReLU())
    layers.append(Affine(params['w2'], params['b2']))
    #layers.append(SoftmaxWithLoss())


def forward_propagation(x):
    for layer in layers:
        x = layer.forward(x)
        return x

def loss(x):                        #softmax
    y = forward_propagation(x)
    return y


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

