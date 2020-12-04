# Wine Binary Classification Model(와인 종류 분류 모델)
# Model fitting


import pandas as pd
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

# 1. load training/test data


dataset_file = './dataset/wine.csv'
df = pd.read_csv(dataset_file, header=None)
df = df.sample(frac=1)
# print(df.info())
# print(df.head())

# 데이터 분류
dataset = df.values
x = dataset[:, 0:12]
t = dataset[:, 12]

# print(x.shape)
# print(y.shape)

# One-Hot 만들기
t = t[:, np.newaxis]
t = np.c_[t, t==0]


# 2. model frame config
model = Sequential()
model.add(Dense(30, input_dim=x.shape[1], activation='relu'))
model.add(Dense(16,  activation='relu'))
model.add(Dense(8,  activation='relu'))
model.add(Dense(2,  activation='softmax'))

# 3. model fitting config
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])


# 4. model fitting
model.fit(x, t, epochs=200, batch_size=100, verbose=1)

# 5. result
result = model.evaluate(x, t, verbose=0)
print(f'\n(Loss, Accuracy) = ({result[0], result[1]}')


# 6. predict
data = np.array([[5.9, 0.26, 0.21, 12.5, 0.034,36, 152, 0.9972, 3.28, 0.43, 9.5, 6]])

predict = model.predict(data)
predict = model.predict(data)
index = np.argmax(predict)
wines = ['Red Wine', 'White Wine']
print(wines[index])




