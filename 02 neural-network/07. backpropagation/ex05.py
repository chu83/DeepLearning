# Sigmoid Layer Test

import numpy as np

import os
import sys
from pathlib import Path

try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from layers import SoftmaxWithLoss
    from common import softmax
    import twolayernet as network

except ImportError:
    print('Library Module Can Not Found')

# 1. Load training/tet data
_x, _t = np.array([2.6, 3.9, 5.6]), np.array([0, 0, 1])

# 2. hyperparameter

# 3. initialize network
layer = SoftmaxWithLoss()


# Test

loss = layer.forward(_x, _t)
dout = layer.backward(1)
print(loss, dout)

#===============================

def forward_propagation(x):
    y = softmax(x)
    return y

network.forward_propagation = forward_propagation
loss = network.loss(_x, _t)
print(loss)



































