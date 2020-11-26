import pickle
import time

import numpy as np
import os
import sys
from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image


try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
    import twolayernet as network
except ImportError:
    print('Library Module Can Not Fount')

# 1. load training / test data
(train_x, train_t), (test_t, test_t) = load_mnist(normalize = True, flatten = True, one_hot_label = True)
# x = np.array([
#     [0.6, 0.9],
#     [0.6, 0.9],
#     [0.6, 0.9, 0.11]
#
#     ])                          # 입력(x)              2 * 2 matrix
# t = np.array([
#     [0., 0., 1.],
#     [0., 1., 0.],
#     [0.6, 0.9, 0.11]
# ])                              # label(one-hot)      2 * 3 vector

# 2. hyperparameters
numiters = 1    #10000
szbatch = 100
sztrain = train_x.shape[0]
ratelearning  = 0.1

# 3. initialize network
network.initialize(sz_input = train_x.shape[1], sz_hidden = 3, sz_output = train_t.shape[1])
# print(network.params['w1'].shape)
# print(network.params['b1'].shape)
# print(network.params['w2'].shape)
# print(network.params['b2'].shape)


# 4. training
train_losses =[]
for idx in range(numiters):
    start = time.time()
    print(f'start')
    # 4-1. fetch mini-batch
    batch_mask = np.random.choice(sztrain, szbatch)
    #print(batch_mask)
    train_x_batch = train_x[batch_mask]
    train_t_batch = train_t[batch_mask]

    #print(train_x_batch.shape)


    # 4-2. gradient
    gradient = network.numerical_gradient_net(train_x, train_t)

    # 4-3. update parameter
    for key in network.params:
        network.params[key] -= ratelearning * gradient[key]

    # 4-4. train loss
    loss = network.loss(train_x_batch, train_t_batch)
    train_losses.append(loss)

    # stopwatch : start
    end = time.time()
    print(f'#{idx+1} : loss :{loss}, elapsed time : {end-start}s')

#serialize train loss
train_loss_file = os.path.join(os.getcwd(), 'dataset', 'twolayer-train-loss.pkl')
print(f'Save Pickle({train_loss_file} file...')
with open(train_loss_file, 'wb') as f:
    pickle.dump(train_losses, f, -1)

print('Done')









#gradient = network.numerical_gradient_net(x, t)

#print(gradient)