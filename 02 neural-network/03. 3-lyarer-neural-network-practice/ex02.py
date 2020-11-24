#3층 신경망 신경 전달 구현2 : 은닉 1층 활성함수 h() 적용
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import sigmoid
    from ex01 import a1
except ImportError:
    print('Library Module Can Not Fount')


print('\n = 신호 전달 구현 2 : 은닉 1층 활성함수 h() 적용========= ')
print(f'a1 dimension : {a1.shape}')  #3vector

z1 = sigmoid(a1)
print(f'z1 = {z1}')