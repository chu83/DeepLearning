#3층 신경망 신경 전달 구현 4 : 은닉 2층 활성함수 h() 적용
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
from pathlib import Path
try:
    sys.path.append(os.path.join(os.getcwd()))
    from ex03 import a2
    from common import sigmoid
except ImportError:
    print('Library Module Can Not Fount')


print('\n = 신호 전달 구현 5 : 은닉 2층 활성함수 h() 적용========= ')
print(f'a2 dimension : {a2.shape}')  # 2 vector

z2 = sigmoid(a2)
print(f'z2 = {z2}')

