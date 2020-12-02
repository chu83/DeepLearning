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
        return y.T

    x = x - np.max(x)
    y  = np.exp(x) / np.sum(np.exp(x))
    return y

# Sum of Squares Error(SSE)
def sum_squares_error(y,t):
    e = 0.5 * np.sum((y-t)**2)
    return e


# Cross entropy error
# t = one hot

def cross_entropy_error_non_batch(y, t):
    delta = 1.e-7
    e = -np.sum(t * np.log(y + delta))
    return e



# Cross entropy error
# t = one hot
# for batch
def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    #print(y.shape)
    batch_size = y.shape[0]
    #print(batch_size)

    delta = 1.e-7
    e = -np.sum(t * np.log(y+delta)) / batch_size
    return e
#
# def numerical_gradient1(f, w, x, t):
#
#     """
#     return 변수 x (벡터, 1차원 numpy array)에 대한 편미분 결과(벡터, 1차원 numpy array) 반환
#
#     :param f: 손실함수
#     :param w:
#     :param x:
#     :param t:
#     :return:
#     """
#
#     h = 1e-4
#     dx = np.zeros_like(w)
#
#     for i in range(w.size):
#         tmp = w[i]
#
#         w[i] = tmp + h
#         h1 = f(w, x, t)
#
#         w[i] = tmp - h
#         h2 = f(w, x, t)
#
#         dx[i] = (h1 - h2) / (2 * h)
#         w[i] = tmp
#
#     return dx
#

def numerical_diff1(f, w, x, t):
    h = 1e-4
    gradient = np.zeros_like(w)

    it = np.nditer(w, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp = w[idx]

        w[idx] = tmp + h
        h1 = f(w, x, t)

        w[idx] = tmp - h
        h2 = f(w, x, t)

        gradient[idx] = (h1 - h2) / (2*h)

        w[idx] = tmp        #값 복원 시키기

        it.iternext()

    return gradient

numerical_gradient1 = numerical_diff1


def numerical_diff2(f, w):
    h = 1e-4
    gradient = np.zeros_like(w)

    it = np.nditer(w, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp = w[idx]

        w[idx] = tmp + h
        h1 = f(w)

        w[idx] = tmp -h
        h2 = f(w)

        gradient[idx] = (h1 - h2) / (2*h)

        w[idx] = tmp

        it.iternext()

    return gradient

numerical_gradient2 = numerical_diff2






