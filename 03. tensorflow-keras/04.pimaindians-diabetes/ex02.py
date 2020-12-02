# pimaindians diabetes model fitting
# tensor-keras

import numpy as np

# 1. load training/test data
dataset = np.loadtxt('./dataset/pimaindians-diabetes.csv', delimiter=',')

train_x = np.array(dataset[:, 0:8])
train_t = np.array(dataset[:, 8])