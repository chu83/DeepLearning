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
    from common import softmax, cross_entropy_error, numerical_gradient1

except ImportError:
    print('Library Module Can Not Fount')

x = np.array([0.6, 0.9])
t = np.array([0., 0., 1.])      #label(one-hot)

def loss(w, x, t):      #softmax
    a = np.dot(x, w)
    y = softmax(a)
    e = cross_entropy_error(y, t)

    return e


_x = np.array([0.6, 0.9])       # 입력(x)              2 vector
_t = np.array([0., 0., 1.])     # label(one-hot)      3 vector
#_w = np.random.randn(2, 3)     # weight              2*3 vector
_w = np.array([
    [0.02, 0.224, 0.135],
    [0.01, 0.052, 0.345]
])                              # weight,             2*3 matrix

g = numerical_gradient1(loss, _w, _x, _t)

print(g)


# params = {
#     "w1" :
#     "b1" :
#     "w2" :
#     "b2" :
# }
#
# gradient_descent(x, t))
#
#
#
#











