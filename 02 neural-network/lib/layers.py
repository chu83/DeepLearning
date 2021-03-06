
# Sigmoid layer
import numpy as np
import os
import sys
from pathlib import Path

try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import softmax, cross_entropy_error, sigmoid

except ImportError:
    print('Library Module Can Not Found')

# Affine Layer
class Affine:
    def __init__(self, w, b):
        self.x = None
        self.w = w
        self.b = b

        self.dw = None
        self.db = None

    def forward(self, x):
        # 역전파시 전치행렬에 대한 내적을 대비
        if x.ndim == 1:
            x = x[np.newaxis, :]

        self.x = x
        out = np.dot(self.x, self.w) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx



# SoftmaxWithLoss Layer
class SoftmaxWithLoss:
    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, x, t=None):
        self.y = softmax(x)
        self.t = t

        if self.t is None:
            return self.y

        #역전파시 전치행렬에 대한 내적을 대비
        if self.t.ndim == 1:
            self.t = self.t[np.newaxis, :]

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, x):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) / batch_size

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1. - self.out) * self.out

        return dx

# ReLU Layer
class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)

        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0

        dx = dout
        return dx

#=========================================

# Multiply layer

class Multiply:
    def __init__(self):     # 초기화
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y

        out = x * y
        return out

    def backward(self, dout):
        dx = self.y * dout
        dy = self.x * dout

        return dx, dy

class Add:
    def __init__(self):     # 초기화
        pass

    def forward(self, x, y):

        out = x + y
        return out

    def backward(self, dout):
        # dx = dout - self.y
        # dy = dout - self.x

        dx = dout * 1
        dy = dout * 1

        return dx, dy