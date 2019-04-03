import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd;
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import warnings

def loadData(iris):
    X=iris.data
    y=iris.target
    return X,y
# FCM
if __name__ == "__main__":
    x_data_raw,y = loadData(datasets.load_iris());
    x_data = toMatrix(x_data_raw)
    # loss = np.sum(np.multiply(get_membership(x_data), distance_list))
    train(x_data, x_data_raw)
    # computedCenter()
    
        