import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# read data
data = pd.read_csv('Train data.csv')

# split data into train data and test data
x = data.iloc[:, : -1]
y = data.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_size=0)
