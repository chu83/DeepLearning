# 신경망학습 : 신경망에서의 기울기


import numpy as np
import os
import sys
from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image

try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
    from common import softmax, cross_entropy_error, numerical_gradient2

except ImportError:
    print('Library Module Can Not Fount')


x = np.array([0.6, 0.9])        # 입력(x)              2 vector
t = np.array([0., 0., 1.])      # label(one-hot)      3 vector

params = {
    'w1' : np.array([[0.02, 0.224, 0.135], [0.01, 0.052, 0.345]]),
    'b1' :np.array([0.45, 0.23, 0.11])
}

def forward_progation():
    w1 = params['w1']
    b1 = params['b1']

    a = np.dot(x, w1) + b1
    y = softmax(a)
    return y                         #softmax(x @ w)

def loss():                        #softmax
    y = forward_progation()
    e = cross_entropy_error(y, t)

    return e

def numerical_gradient_net():
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
            h1 = loss()

            param[idx] = tmp - h
            h2 = loss()

            param_gradient[idx] = (h1 - h2) / (2 * h)

            param[idx] = tmp

            it.iternext()
        gradient[key] = param_gradient

    return gradient

g= numerical_gradient_net()
print(g)

