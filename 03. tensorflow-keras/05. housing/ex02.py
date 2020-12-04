# Housing Mineral Binary Classification Model
# Model Fitting(학습)


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

# 1. load training/test data


dataset_file = './dataset/housing.csv'
df = pd.read_csv(dataset_file, delim_whitespace=True, header = None)

dataset = df.values
x = dataset[:, 0:13]
t = dataset[:, 13]

print(x.shape)
print(t.shape)
train_x, test_x, train_t, test_t = train_test_split(x, t, test_size=0.3, random_state=0)

# 2. model frame config
model = Sequential()
model.add(Dense(30, input_dim=train_x.shape[1], activation='relu'))
model.add(Dense(10,  activation='relu'))
model.add(Dense(1))


# 3. model fitting config
model.compile(loss='mean_squared_error', optimizer='adam')


# 4. model fitting
model.fit(train_x, train_t, epochs=200, batch_size=10)

# 5. prediction test
y = model.predict(test_x)
#print(y)
for i in range(len(test_t)):
    label = test_t[i]
    prediction = y[i]
    print(f'실제가격:{label}, 예상가격:{prediction:.3f}')













