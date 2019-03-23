import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import warnings


if __name__ == "__main__":
    X_data,y = loadData(datasets.load_iris());
    print(X_data)