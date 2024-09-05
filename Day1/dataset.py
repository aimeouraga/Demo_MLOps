import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

iris=load_iris()
for keys in iris.keys() :
    print(keys)

X = iris.data
y = iris.target

print(X[0:4])
print(y[0:4])
