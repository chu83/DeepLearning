# Sonar Mineral binary Classification Model(초음파 광물 예측 모델)
# Model Fitting(학습)


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

# 1. load training/test data


dataset_file = 'dataset/sonar.csv'
df = pd.read_csv(dataset_file, header = None)

dataset = df.values
x, t = dataset[:, 0:60].astype(float), dataset[:, 60]

e = LabelEncoder()
e.fit(t)
t = e.transform(t)
t = tf.keras.utils.to_categorical(t)
print(t)

# 2. model frame config
model = Sequential()
model.add(Dense(20, input_dim=60, activation='relu'))
model.add(Dense(10, input_dim=60, activation='relu'))
model.add(Dense(2, input_dim=10, activation='softmax'))
#model.add(Dense(1, input_dim=10, activation='sigmomid'))


# 3. model fitting config
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. model fitting
history = model.fit(x, t, epochs=100, batch_size=5, verbose=1)


# 5. result
loss = history.history['loss']
result = model.evaluate(x, t, verbose=0)
print(f'\n(Loss, Accuracy) = ({result[0], result[1]}')













