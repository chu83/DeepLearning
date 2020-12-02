# Training Neural Network
# Data Set : MNIST Handwritten Digit Dataset
# Network : TwoLayerNet
# Test : TwoLayerNet
# Estimation : Training Accuracy


import pickle
import time

import numpy as np
import os
import sys
from pathlib import Path

try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
    import twolayernet2 as network
except ImportError:
    print('Library Module Can Not Fount')


train_loss_file = os.path.join(os.getcwd(), 'dataset', f'twolayer_train_loss.pkl')
train_losses = None


with open(params_file, 'wb') as f_params,\
        open(train_loss_file, 'rb') as f_trainloss,\
        open(trainacc_file, 'wb') as f_trainacc,\
        open(testacc_file, 'wb') as f_testacc:
    pickle.dump(network.params, f_params, -1)
    pickle.dump(train_losses, f_trainloss, -1)
    pickle.dump(train_accuracies, f_trainacc, -1)
    pickle.dump(test_accuracies, f_testacc, -1)
















