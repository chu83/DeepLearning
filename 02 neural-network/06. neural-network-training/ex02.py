# 신경망학습 : 오차제곱합 손실함수 (Cross Entropy Error , CEE)


import numpy as np
import os
import sys
from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image

try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import cross_entropy_error

except ImportError:
    print('Library Module Can Not Fount')


t = np.array( [0., 0., 1., 0., 0., 0., 0., 0., 0., 0. ])

y1 = np.array([0.1, 0.05, 0.7, 0., 0.02, 0.03, 0.1, 0., 0., 0.])
y2 = np.array([0.1, 0.05, 0.05, 0.4, 0.02, 0.03, 0.1, 0.3, 0., 0.])
y3 = np.array([0., 0.92, 0.02, 0., 0.02, 0.03, 0.1, 0., 0., 0.])


#test
# print(cross_entropy_error_non_batch(y1, t))
# print(cross_entropy_error_non_batch(y2, t))
# print(cross_entropy_error_non_batch(y3, t))

#test
print(cross_entropy_error(y1, t))
print(cross_entropy_error(y2, t))
print(cross_entropy_error(y3, t))