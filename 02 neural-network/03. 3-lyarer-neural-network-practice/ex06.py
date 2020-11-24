#3층 신경망 신경 전달 구현 6 : 출력층 출력함수 sigma() 적용
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
from pathlib import Path
try:
    sys.path.append(os.path.join(os.getcwd()))
    from ex05 import a3
    from common import identity
except ImportError:
    print('Library Module Can Not Fount')


print('\n = 3층 신경망 신경 전달 구현 6 : 출력층 출력함수 sigma() 적용========= ')
print(f'a3 dimension : {a3.shape}')  # 2 vector

y = identity(a3)
print(f'y = {y}')