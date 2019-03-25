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
# 生产随机中心点
def getRamdonCenter(k, x_data):
    # mininum = np.min(x_data)
    # maxinum = np.max(x_data)
    maxList = np.argmax(x_data, 0)
    maxList = tf.map_fn(elems= maxList, fn=lambda index: 1) # x_data[maxList[index]][index]
    return maxList
    # minList = np.argmin(x_data, 0)
    # minList = tf.map_fn(elems=minList, fn=lambda index: x_data[minList[index]][index])
    # res = []
    # for i in range(k):
    #     # for mi, ma in zip(np.array(minList), np.array(maxList)):
    #         # print(mi, ma)
    #     res.append(list(np.random.uniform(low=mi, high=ma) for mi, ma in zip(np.array(minList), np.array(maxList))))
    # return np.array(res)


def formate(item, mininum, maxinum):
    res = (item - mininum) / (maxinum - mininum)
    return res 

def toMatrix(x_data):
    # x_data = np.matrix(x_data)
    mininum = np.min(x_data)
    maxinum = np.max(x_data)
    # 遍历矩阵 规整化
    for RowIndex, RowItem in enumerate(x_data):
        for ColIndex, ColItemm in enumerate(RowItem): 
            RowItem[ColIndex] = formate(ColItemm, mininum, maxinum)
        x_data[RowIndex] = RowItem
    return x_data

# 两点距离
def distance(x1_list, x2_list):
    sum = 0
    for x1, x2 in zip(x1_list, x2_list):
        sum += (x2 - x1) ** 2
    sum = sum ** 0.5
    return sum
def get_membership(x_data):
    membership = np.random.random(size=(3, len(x_data)))
    return membership

def draw(data):
    # data = pd.DataFrame(data)
    fig = plt.figure() 
    ax = Axes3D(fig) 
    ax.scatter(data[:, 0], data[:, 2], data[:, 1])

def train(x_data):
    with tf.Session() as sess:
        x_data_tf = tf.placeholder(shape=[None, 4], dtype=tf.float32)
        centerList = getRamdonCenter(3, x_data_tf)
        # distance_list = list(list(distance(x, i) for x in x_data) for i in centerList)
        # loss = tf.sum(tf.multiply(get_membership(x_data_tf), distance_list))
        # train_step = my_opt.minimize(loss)
        # 训练开始
        init = tf.global_variables_initializer()
        sess.run(init)
        print(sess.run(centerList, feed_dict={x_data_tf: x_data}))
        # sess.run(centerList, feed_dict={x_data_tf: x_data})
# FCM
if __name__ == "__main__":
    x_data,y = loadData(datasets.load_iris());
    draw(x_data)
    x_data = toMatrix(x_data)
    # loss = np.sum(np.multiply(get_membership(x_data), distance_list))
    train(x_data)
        