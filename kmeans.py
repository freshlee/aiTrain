import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import warnings
#warnings.filterwarnings("ignore")

def loadData(iris):
    X=iris.data
    y=iris.target
    print(X)
    return X,y
def kmeansCluster(X,numClusters):
    get_inputs = lambda: tf.train.limit_epochs(tf.convert_to_tensor(X, dtype=tf.float32), num_epochs=1)
    # print(sess.run(get_inputs()), 'get_inputs')
    cluster = tf.contrib.factorization.KMeansClustering(num_clusters=numClusters, initial_clusters=tf.contrib.factorization.KMeansClustering.KMEANS_PLUS_PLUS_INIT)
    cluster.train(input_fn=get_inputs, steps=2000)
    y_pred=cluster.predict_cluster_index(input_fn=get_inputs)
    y_pred=np.asarray(list(y_pred))
    return y_pred
def plotFigure(fignum,title, X,y):
    fig = plt.figure(fignum, figsize=(8,6))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y.astype(np.float), edgecolor='k')
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    ax.set_title(title)
    ax.dist = 10
    fig.show()


if __name__ == '__main__':
    # sess = tf.Session()
    X,y = loadData(datasets.load_iris())
    y_pred = kmeansCluster(X,1)
    plotFigure(1,"3 clusters",X,y_pred)
    plotFigure(2,"Ground Truth",X,y)