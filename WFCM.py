import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import warnings


# 生产随机中心点
def getRamdonCenter(k, x_data):
    # mininum = np.min(x_data)
    # maxinum = np.max(x_data)
    maxList = np.argmax(x_data, 0)
    maxList = list(x_data[item][index] for index, item in enumerate(maxList))
    minList = np.argmin(x_data, 0)
    minList = list(x_data[item][index] for index, item in enumerate(minList))
    print(maxList, minList)
    res = []
    for i in range(k):
        # for mi, ma in zip(np.array(minList), np.array(maxList)):
            # print(mi, ma)
        res.append(list(np.random.uniform(low=mi, high=ma) for mi, ma in zip(np.array(minList), np.array(maxList))))
    print(res)


def formate(item, mininum, maxinum):
    res = (item - mininum) / (maxinum - mininum)
    return res 

def toMatrix(x_data):
    x_data = np.matrix(x_data)
    mininum = np.min(x_data)
    maxinum = np.max(x_data)
    # 遍历矩阵 规整化
    for RowIndex, RowItem in enumerate(x_data):
        for ColIndex, ColItemm in enumerate(RowItem): 
            RowItem[ColIndex] = formate(ColItemm, mininum, maxinum)
        x_data[RowIndex] = RowItem
    print(x_data)
def distance(x1_list, x2_list):
    for x1, x2 in zip(x1_list, x2_list):
        res = np.multiply(np.matrix(x1) - np.matrix(x2))
# FCM
if __name__ == "__main__":
    X_data,y = loadData(datasets.load_iris());
    # toMatrix(X_data)
    getRamdonCenter(3, X_data)