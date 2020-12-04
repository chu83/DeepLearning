# Housing Mineral Binary Classification Model
# Model fitting - dl class version
import time

import pandas as pd
import os
import sys
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
    import multilayernet as network
except ImportError:
    print('Library Module Can Not Fount')


# 1. load training/test data


dataset_file = './dataset/sonar.csv'
df = pd.read_csv(dataset_file, header = None)

dataset = df.values

x = dataset[:, 0:13]
t = dataset[:, 13]

train_x, test_x, train_t, test_t = train_test_split(x, t, test_size=0.3, random_state=0)


# 2. model frame config
model = Sequential()
model.add(Dense(20, input_dim=60, activation='relu'))
model.add(Dense(10, input_dim=60, activation='relu'))
model.add(Dense(2, input_dim=10, activation='softmax'))

# 2. hyperparameters

batch_size = 100
epochs = 20
learning_rate  = 0.1

# 3. Model frame
network.initialize(input_size = train_x.shape[1], hidden_size = [50, 100], output_size = train_t.shape[1])


# 4. Model fitting
train_size = train_x.shape[0]
epoch_size = int(train_size / batch_size)
iterations = epochs * epoch_size

elapsed = 0
epoch_idx = 0


for idx in range(1, iterations+1):

    # 4-1. fetch mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    train_x_batch = train_x[batch_mask]             #100*784
    train_t_batch = train_t[batch_mask]             #100*10

    #print(train_x_batch.shape)


    # 4-2. gradient
    start = time.time()

    gradient = network.backpropagation_gradient_net(train_x_batch, train_t_batch)
    elapsed += (time.time() - start)

    # 4-3. update parameter
    for key in network.params:
        network.params[key] -= learning_rate * gradient[key]

    # 4-4. train loss
    loss = network.loss(train_x_batch, train_t_batch)
    train_losses.append(loss)

    # 4-5 accuracy per epoch
    if idx % epoch_size == 0:
        epoch_idx += 1
        train_accuracy = network.accuracy(train_x, train_t)
        train_accuracies.append(train_accuracy)

        test_accuracy = network.accuracy(test_x, test_t)
        test_accuracies.append(test_accuracy)

        print(f'\nEpoch {epoch_idx:02d}/{epochs:02d}')

        print(f'{int(idx/epoch_size)}/{epoch_size}: - elapsed time : {elapsed*1000:.3f}ms - loss :{loss:.3f}, ')

        elapsed = 0