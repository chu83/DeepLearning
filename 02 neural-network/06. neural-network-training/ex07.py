# Training Neural Network
# Data Set: MNIST Handwritten Digit Dataset
# Network: TwoLayerNet
# Estimation: Training Accuracy

import os
import pickle
import sys
from pathlib import Path
from matplotlib import pyplot as plt

try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
    import twolayernet as network
except ImportError:
    print('Library Module Can Not Found')


trainacc_file = os.path.join(os.getcwd(), 'dataset', f'twolayer_train_accuracy.pkl')
testacc_file = os.path.join(os.getcwd(), 'dataset', f'twolayer_test_accuracy.pkl')

train_accuracies = None
test_accuracies = None


with open(trainacc_file, 'rb') as f_trainacc, open(testacc_file, 'rb') as f_testacc:
    train_accuracies = pickle.load(f_trainacc)
    test_accuracies = pickle.load(f_testacc)


plt.plot(train_accuracies, label = 'Train Accuracy')
plt.plot(test_accuracies, label = 'Test Accuracy')

plt.xlim(0, 20, 1)
plt.ylim(0, 1.0, 0.5)

plt.xlabel('Epocy')
plt.ylabel('Accuracy')
plt.legend(loc = 'best')

plt.show()