# MNIST(Modified National Institude of Standard and Technology)
# MNIST 손글씨 숫자 분류 신경망(Neural Network for MNIST Handwritten Digit Classification : 데이터 살펴뵈기

import numpy as np
import os
import sys
from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image

try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist

except ImportError:
    print('Library Module Can Not Fount')

(train_x, train_t), (test_x, test_t) = load_mnist(normalize=False, flatten=True, one_hot_label = False)

print(train_x.shape)    #60,000*784 matrix
print(train_t.shape)    #60,000 * 10 vector

t = train_t[0]
print(t)                # 5

x = train_x[0]
print(x)
print(x.shape)          # 784
x = x.reshape(28, 28)   # 형상을 원래 이미지 크기로 변경
print(x.shape)          # 28*28

# 이미지 보기 : PIL(Python Image Library) 사용
pil_Image = Image.fromarray(x)
pil_Image.show()














