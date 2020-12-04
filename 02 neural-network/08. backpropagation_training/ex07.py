# Training Neural Network
# Data Set : MNIST Handwritten Digit Dataset
# Network : TwoLayerNet
# Test : TwoLayerNet
# Estimation : Training

from matplotlib import pyplot as plt
import pickle
import time

import numpy as np
import os
import sys
from pathlib import Path

try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
    import multilayernet as network
except ImportError:
    print('Library Module Can Not Fount')


trainacc_file = os.path.join(os.getcwd(), 'model', f'twolayer_trainacc.pkl')
testacc_file = os.path.join(os.getcwd(), 'model', f'twolayer_testacc.pkl')

train_accuracies = None
test_accuracies = None


with open(trainacc_file, 'rb') as f_trainacc,\
        open(testacc_file, 'rb') as f_testacc:
    train_accuracies =  pickle.load(f_trainacc)
    test_accuracies = pickle.load(f_testacc)

xlen = np.arange(len(train_accuracies))

plt.plot(xlen, train_accuracies, marker = '.', c = 'blue', label = 'train accuracy')
plt.plot(xlen, test_accuracies, marker = '.', c = 'red', label = 'test accuracy')

plt.ylim(0.8, 1., 0.5)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='best')

plt.show()














