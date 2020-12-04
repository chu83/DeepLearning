# Training Neural Network
# Data Set : MNIST Handwritten Digit Dataset
# Network : TwoLayerNet
# Test : TwoLayerNet
# Estimation : Training Accuracy


import pickle
from matplotlib import pyplot as plt

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


train_loss_file = os.path.join(os.getcwd(), 'model', f'twolayer_train_loss.pkl')
train_losses = None


with open(train_loss_file, 'rb') as f:     # 'rb' : 읽기모드, 'wb' : 쓰기모드
    train_losses = pickle.load(f)

plt.plot(train_losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')

plt.show()














