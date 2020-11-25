# MNIST 손글씨 숫자 분류 신경망(Neural Network for MNIST Handwritten Digit Classification) : 신호전달 1


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

print(w1.shape)         # 784*50 matrix
print(w2.shape)         # 50*100 matrix
print(w3.shape)         # 100*10 matrix

print(b1.shape)         # 50 vector
print(b2.shape)         # 100 vector
print(b3.shape)         # 10 vector

# 2 학습/시험 데이터 가져오기
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label = False)


xlen = len(test_x)
randidx = np.random.randint(0, xlen, 1).reshape(())
print(randidx)


# 3. 신호전달
print('\n == 신호전달 구현 1 : 은닉 1층 전달 =============================')
x = test_x[randidx]
print(f'x dimension : {x.shape}')

w1 = network['W1']
print(f'w1 dimension : {w1.shape}')

b1 = network['b1']
print(f'b1 dimension : {b1.shape}')

a1 = np.dot(x, w1) +b1
print(f'a1 = {a1}')



print('\n == 신호전달 구현 2 : 은닉 1층 활성함수 h() 적용 =============================')
print(f'a2 dimension : {a1.shape}') #100 vector
z1 = sigmoid(a1)
print(f'z2 = {z1}')




print('\n == 신호전달 구현 3 : 은닉 2층 전달 =============================')
x = test_x[randidx]
print(f'x dimension : {x.shape}')

w2 = network['W2']
print(f'w2 dimension : {w2.shape}')

b2 = network['b2']
print(f'b2 dimension : {b2.shape}')

a2 = np.dot(z1, w2) +b2
print(f'a2 = {a2}')



print('\n == 신호전달 구현 4 : 은닉 1층 활성함수 h() 적용 =============================')
print(f'a2 dimension : {a2.shape}') #100 vector
z2 = sigmoid(a2)
print(f'z2 = {z2}')


print('\n == 신호전달 구현 5 :  출력층 전달 =============================')
print(f'z2 dimension : {z2.shape}')     # 100 vector
w3 = network['W3']
print(f'w3 dimension : {w3.shape}')     # 100*10 matrix
b3 = network['b3']
print(f'b3 dimension : {b3.shape}')     # 10 vector
a3 = np.dot(z2, w3) + b3
print(f'a3 = {a3}')


print('\n == 신호전달 구현 6 : 출력층 활성함수 sigma() 적용 =============================')
print(f'a3 dimension : {a3.shape}')     # 10 vector
y = softmax(a3)
print(y, np.sum(y))


print('\n == 예측결과 =============================')
predict = np.argmax(y)
print(f'{randidx + 1} 번째 이미지 예측 : {predict}')


print('\n == 정답 =============================')
t = test_t[randidx]
print(f'{randidx+1} 번째 이미지 레이블 : {t}')













