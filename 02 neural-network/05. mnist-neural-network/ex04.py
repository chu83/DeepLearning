# MNIST 손글씨 숫자 분류

import numpy as np
import os
import sys
from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image

try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import init_network, load_mnist
    from common import sigmoid, softmax

except ImportError:
    print('Library Module Can Not Fount')

# 1. 매개변수(w,b) 데이터 셋 가져오기
network = init_network()

w1, w2, w3 = network['W1'], network['W2'], network['W3']
b1, b2, b3 = network['b1'], network['b2'], network['b3']


# 2 학습/시험 데이터 가져오기
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label = False)

xlen = len(test_x)


# 3. 정확도 산출
hit = 0
for idx in range(xlen):
    x = test_x[idx]

    a1 = np.dot(x, w1) +b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, w2) +b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    predict = np.argmax(y)
    t = test_t[idx]

    if predict == t:
        hit += 1


# 정확도 (Accuracy)
print(f'Accuracy : {hit/xlen}')




