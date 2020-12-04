# Housing Mineral Binary Classification Model
# Explore Dataset(데이터탐색)

import pandas as pd

import pandas as pd


dataset_file = './dataset/housing.csv'
df = pd.read_csv(dataset_file, header=None)
print(df.info())
print(df.head())