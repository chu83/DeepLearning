# nand gate : perceptron
import numpy as np
import os
import sys

from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()), 'lib'))
    from common import step
except ImportError:
    print('Library Module Can Not Fount')

def OR(x):
    w, b = np.array([0.5, 0.5]), -0.2
    a = np.sum(x * w) + b
    y = step(a)

    return y


if __name__ == '__main__':
    y1 = OR(np.array([0, 0]))
    print(y1)
    y2 = OR(np.array([0, 1]))
    print(y2)
    y3 = OR(np.array([1, 0]))
    print(y3)
    y4 = OR(np.array([1, 1]))
    print(y4)