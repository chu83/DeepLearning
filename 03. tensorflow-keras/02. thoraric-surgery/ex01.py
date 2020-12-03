import pandas as pd
import numpy as np

dataset_file = './dataset/thoraric-surgery.csv'

df = pd.read_csv(dataset_file, header=None, delim_whitespace=True)

print(df.info)
print(df.head())

# 데이터 분리하기
dataset = np.loadtxt(dataset_file, delimiter=',')
print(dataset.shape)

x = np.array(dataset[:, 0:17])
t = np.array(dataset[:, 17])

print(x.shape, t.shape)


# Config. Model Frame
# input_dim = 17
# hidden layer1 : 17*30, relu
# output layer : 1, sigmoid


# config, Model Fitting
# loss function : binary_crossentropy
#optimize = sgd, adam
# epochs : 100, batch_size : 10
















