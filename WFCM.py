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
    print(x_data)
    maxList = np.argmax(x_data, 0)
    maxList = list(x_data[item][index] for index, item in enumerate(maxList))
    minList = np.argmin(x_data, 0)
    minList = list(x_data[item][index] for index, item in enumerate(minList))
    res = []
    for i in range(k):
        # for mi, ma in zip(np.array(minList), np.array(maxList)):
            # print(mi, ma)
        res.append(list(np.random.uniform(low=mi, high=ma) for mi, ma in zip(np.array(minList), np.array(maxList))))
    return np.array(res)


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
    x_data = x_data[:,[0, 2, 1]]
    # print(x_data)
    return x_data

# 两点距离
def distance(x1_list, x2_list):
    sum = 0 
    for x1, x2 in zip(x1_list, x2_list):
        sum += (x2 - x1) ** 2
    sum = sum ** 0.5
    return sum
def get_membership(x_data):
    membership = tf.random_normal(shape=tf.shape(x_data),dtype=tf.float32)
    membership = tf.Variable(membership)
    return membership

def draw(data):
    # data = pd.DataFrame(data)
    fig = plt.figure() 
    ax = Axes3D(fig) 
    ax.scatter(data[:, 0], data[:, 2], data[:, 1])

def computedCenter(u, x_data):
    # total = 5
    # cate = 3
    # u = np.random.random((cate, total))
    # x_data = np.random.random((total, 3))
    sums = tf.tensordot(u, x_data, axes=0)
    res = tf.divide(sums, tf.reduce_sum(u))
    # print(sums, res)
    return res


def train(x_data):
    centerListInit = getRamdonCenter(3, x_data)
    print(centerListInit)
    with tf.Session() as sess:
        x_data_tf = tf.placeholder(shape=[None, 3], dtype=tf.float32)
        centerList = tf.placeholder(shape=[3, None], dtype=tf.float32)
        # 每个点离分类点的距离[3, N]
        centerList=centerList.eval(feed_dict={x_data_tf: x_data, centerList: centerListInit})
        distance_list = list(list(distance(x, i) for x in x_data) for i in centerList)
        distance_list=tf.convert_to_tensor(distance_list)
        # u = get_membership(x_data)
        # centerList = computedCenter(u, x_data_tf)
        loss = tf.reduce_mean(tf.multiply(centerList, tf.transpose(distance_list)))
        my_opt = tf.train.GradientDescentOptimizer(0.01)
        train_step = my_opt.minimize(loss)
        # 训练开始
        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(train_step, feed_dict={x_data_tf: x_data, centerList: centerListInit})
        # print(centerList)
        # sess.run(centerList, feed_dict={x_data_tf: x_data})
# FCM
if __name__ == "__main__":
    x_data,y = loadData(datasets.load_iris());
    draw(x_data)
    x_data = toMatrix(x_data)
    # loss = np.sum(np.multiply(get_membership(x_data), distance_list))
    train(x_data)
    # computedCenter()
    
        