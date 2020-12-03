# Sonar Mineral binary Classification Model(초음파 광물 예측 모델)
# Model Fitting #3 - K-Fold Cross Validation

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import load_model


# 1-1. load training/test data

dataset_file = './dataset/sonar.csv'
df = pd.read_csv(dataset_file, header = None)

dataset = df.values
x, t = dataset[:, 0:60].astype(float), dataset[:, 60]

e = LabelEncoder()
e.fit(t)
t = e.transform(t)
#t = tf.keras.utils.to_categorical(t)
#print(t)

# 1-2 10-foldder Crross Validation
nfold = 10
skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=0)
accuracies = []


for mask_train, mask_test in skf.split(x, t):
    # x[train]
    # print(train)

    # 2. model frame config
    model = Sequential()
    model.add(Dense(20, input_dim=60, activation='relu'))
    model.add(Dense(10, input_dim=60, activation='relu'))
    model.add(Dense(1, input_dim=10, activation='sigmoid'))

    # 3. model fitting config
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


    # 4. model fitting
    #history = model.fit(x, t, epochs=100, batch_size=5, verbose=1)
    history = model.fit(x[mask_train], t[mask_train], epochs=100, batch_size=5, verbose=1)


    # 5. result
    result = model.evaluate(x[mask_test], t[mask_test], verbose=0)
    accuracies.append(result[1])

    #loss = history.history['loss']
    #result = model.evaluate(x, t, verbose=0)

    print(f'\n{nfold} fold accuracies : {accuracies}')
    #print(f'\n(Loss, Accuracy) = ({result[0], result[1]}')
    #print(f'\n(Loss, Accuracy) = ({result[0], result[1]}')

