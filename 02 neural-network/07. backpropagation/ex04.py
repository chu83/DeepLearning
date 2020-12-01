# Sigmoid layer Test
import numpy as np

import os
import sys
from pathlib import Path

try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from layers import Sigmoid

except ImportError:
    print('Library Module Can Not Found')

# Test1(Vector)
layer = Sigmoid()

x = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
print(x)

y = layer.forward(x)
print(y)
print(layer.out)

dout = np.array([-0.1, -0.2, -0.3, 0.4, -0.5])
dout = layer.backward(dout)
print(dout)

print('================================')