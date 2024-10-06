from perceptron import Perceptron
import os
import pandas as pd
import numpy as np

# Read iris data from source
source = os.path.join('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
df = pd.read_csv(source, header=None, encoding='utf-8')

# Prepare data
#
# Take  first 100 rows, 5th column with names
# then create known values for them
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# Create train set, take only first and 3rd property
X = df.iloc[0:100, [0, 2]].values

perceptron_one = Perceptron(0.1, 10, 1)

# train the perceptron
perceptron_one.fit(X, y)