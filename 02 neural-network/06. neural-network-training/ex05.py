# Training Neural Network
# Data Set : MNIST Handwritten Digit Dataset
# Network : TwoLayerNet
# Estimation : Training
import datetime
import pickle
import time

import numpy as np
import os
import sys
from pathlib import Path

try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
    import twolayernet as network
except ImportError:
    print('Library Module Can Not Fount')

# 1. load training / test data
(train_x, train_t), (test_x, test_t) = load_mnist(normalize = True, flatten = True, one_hot_label = True)
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
numiters = 1    #10000  2000
szbatch = 100
sztrain = train_x.shape[0]      # 60,000
szepoch = sztrain / szbatch     # 전체 학습 데이터로 학습을 끝마쳤을 때 --> 1 epoch : 60,000 / 100 = 600
ratelearning  = 0.1

# 3. initialize network
network.initialize(sz_input = train_x.shape[1], sz_hidden = 50, sz_output = train_t.shape[1])
# print(network.params['w1'].shape)
# print(network.params['b1'].shape)
# print(network.params['w2'].shape)
# print(network.params['b2'].shape)


# 4. training
train_losses =[]
train_accuracies = []
test_accuracies = []

for idx in range(numiters+1):
    print(f'start')
    # 4-1. fetch mini-batch
    batch_mask = np.random.choice(sztrain, szbatch)
    #print(batch_mask)
    train_x_batch = train_x[batch_mask]             #100*784
    train_t_batch = train_t[batch_mask]             #100*10

    #print(train_x_batch.shape)


    # 4-2. gradient
    start = time.time()

    gradient = network.numerical_gradient_net(train_x_batch, train_t_batch)
    elapsed = time.time() - start

    # 4-3. update parameter
    for key in network.params:
        network.params[key] -= ratelearning * gradient[key]

    # 4-4. train loss
    loss = network.loss(train_x_batch, train_t_batch)
    train_losses.append(loss)

    # 4-5 : accuracy per epoch
    #if idx / szepoch ==0:
    if True:
        train_accuracy = network.accuracy(train_x, train_t)       # 전수검사
        train_accuracies.append(train_accuracy)

        test_accuracy = network.accuracy(test_x, test_t)
        test_accuracies.append((test_accuracy))

    # stopwatch : start
    end = time.time()
    print(f'#{idx+1} : loss :{loss}, elapsed time : {end-start}s')

# 5. Serialization
now = datetime.datetime.now()

params_file = os.path.join(os.getcwd(), 'dataset', f'twolayer_params_{now:%Y%m%d%H%M%S}.pkl')
train_loss_file = os.path.join(os.getcwd(), 'dataset', f'twolayer_train_loss_{now:%Y%m%d%H%M%S}.pkl')
trainacc_file = os.path.join(os.getcwd(), 'dataset', f'twolayer_trainacc_{now:%Y%m%d%H%M%S}.pkl')
testacc_file = os.path.join(os.getcwd(), 'dataset', f'twolayer_testacc_{now:%Y%m%d%H%M%S}.pkl')

print(f'Creating pickle...')

with open(params_file, 'wb') as f_params,\
        open(train_loss_file, 'wb') as f_trainloss,\
        open(trainacc_file, 'wb') as f_trainacc,\
        open(testacc_file, 'wb') as f_testacc:
    pickle.dump(network.params, f_params, -1)
    pickle.dump(train_losses, f_trainloss, -1)
    pickle.dump(train_accuracies, f_trainacc, -1)
    pickle.dump(test_accuracies, f_testacc, -1)


print('Done')









#gradient = network.numerical_gradient_net(x, t)

#print(gradient)