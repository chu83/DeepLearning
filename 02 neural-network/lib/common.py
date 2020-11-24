import numpy as np

#sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#relu activation function
def relu(x):
    # if x > 0:
    #     return x
    # else:
    #     return 0
    # return x if x > 0 else 0
    return np.maximum(0,x)
def identity(x):
    return x

#softmax activation function : 큰값에서 NAN 변환하는 불안한 함수
def softmax_overflow(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def softmax(x):
    x = x - np.max(x)  #오버플로우
    return np.exp(x) / np.sum(np.exp(x))