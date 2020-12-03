# Iris Species Multi-Class Classification Model(Iris 품종 예측 모델)
# Explore Dataset(데이터 탐색)

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

dataset_file = './dataset/iris.csv'
df = pd.read_csv(dataset_file, names=['sepal length ', 'sepal width', 'petal length', 'petal width', 'species'])
print(df.info())
print(df.head())

# 데이터 분류
dataset = df.values
x, t = dataset[:, 0:4], dataset[:, 4]

# 문자열을 숫자(one-hot) 바꾸기
# [1 0 0] = 'Iris-setosa'
# [0 1 0] = 'Iris-versicolor'
# [0 0 1] = 'Iris-virginica'

t = ['dog', 'cat', 'dog', 'cat']
e = LabelEncoder()
e.fit(t)
t = e.transform(t)
t = tf.keras.utils.to_categorical(t)
print(t)