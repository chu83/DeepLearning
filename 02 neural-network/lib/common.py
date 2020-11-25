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


#softmax activation function : 오버플로우 대책 & 배치처리지원 수정

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis = 0)  #오버플로우
        y = np.exp(x) / np.sum(np.exp(x), axis = 0)

    x = x - np.max(x)
    y  = np.exp(x) / np.sum(np.exp(x))
    return y