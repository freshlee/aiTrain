import numpy as np;
import pandas as pd;
import tensorflow as tf;
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess = tf.Session()
df1 = pd.read_csv('./data/iris.data', sep='\n',low_memory=False,header=None)
trainData, testData = train_test_split(df1, test_size=0.5)
trainData = np.array(df1)
trainData = trainData.tolist()
trainData = [item[0].split(',') for item in trainData]
val_y1 = np.array([1 if item[4] == 'Iris-setosa' else -1 for item in trainData])
val_y2 = np.array([1 if item[4] == 'Iris-versicolor' else -1 for item in trainData])
val_y3 = np.array([1 if item[4] == 'Iris-virginica' else -1 for item in trainData])
val_y = np.array([val_y1, val_y2, val_y3])
val_x = np.array([[float(x[0]), float(x[3])] for x in trainData])
class1_x = [x[0] for i,x in enumerate(val_x) if trainData[i][4]=='Iris-setosa']
class1_y = [x[1] for i,x in enumerate(val_x) if trainData[i][4]=='Iris-setosa']
class2_x = [x[0] for i,x in enumerate(val_x) if trainData[i][4]=='Iris-versicolor']
class2_y = [x[1] for i,x in enumerate(val_x) if trainData[i][4]=='Iris-versicolor']
class3_x = [x[0] for i,x in enumerate(val_x) if trainData[i][4]=='Iris-virginica']
class3_y = [x[1] for i,x in enumerate(val_x) if trainData[i][4]=='Iris-virginica']
batch_size = 50
# 初始化feedin
x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y_target = tf.placeholder(shape=[3, None], dtype=tf.float32)
prediction_grid = tf.placeholder(shape=[None, 2], dtype=tf.float32)

#kernel 核函数只依赖x_data
gamma = tf.constant(-10.0)
sq_dists = tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))
test2 = tf.multiply(gamma, tf.abs(sq_dists))
my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))
# 最大的变化是批量矩阵乘法。
# 最终的结果是三维矩阵，并且需要传播矩阵乘法。
# 所以数据矩阵和目标矩阵需要预处理，比如xT·x操作需额外增加一个维度。
# 这里创建一个函数来扩展矩阵维度，然后进行矩阵转置，
# 接着调用TensorFlow的tf.batch_matmul（）函数
def reshape_matmul(mat):
    print(mat)
    v1 = tf.expand_dims(mat, 1)
    v2 = tf.reshape(v1, [3, batch_size, 1])
    return(tf.matmul(v2, v1))

# 创建变量
b = tf.Variable(tf.random_normal(shape=[3,batch_size]))

first_term = tf.reduce_sum(b)
b_vec_cross = tf.matmul(tf.transpose(b), b)
y_target_cross = reshape_matmul(y_target)
test1 = tf.multiply(b_vec_cross, y_target_cross)
second_term = tf.reduce_sum(tf.multiply(my_kernel,  y_target_cross),[1,2])
loss = tf.reduce_sum(tf.negative(tf.subtract(first_term, second_term)))

# Gaussian (RBF) prediction kernel
# 现在创建预测核函数。
# 要当心reduce_sum（）函数，这里我们并不想聚合三个SVM预测，
# 所以需要通过第二个参数告诉TensorFlow求和哪几个
rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1),[-1,1])
rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1),[-1,1])
pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))), tf.transpose(rB))
pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

# 实现预测核函数后，我们创建预测函数。
# 与二类不同的是，不再对模型输出进行sign（）运算。
# 因为这里实现的是一对多方法，所以预测值是分类器有最大返回值的类别。
# 使用TensorFlow的内建函数argmax（）来实现该功能
prediction_output = tf.matmul(b, pred_kernel)
test = prediction_output-tf.expand_dims(tf.reduce_mean(prediction_output,1), 1)
prediction = tf.arg_max(prediction_output-tf.expand_dims(tf.reduce_mean(prediction_output,1), 1), 0)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y_target,0)), tf.float32))

my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

# 训练开始
loss_vec = []
batch_accuracy = []
for i in range(100):
    rand_index = np.random.choice(len(val_x), size=batch_size)
    rand_x = val_x[rand_index]
    rand_y = val_y[:,rand_index]
    # print(val_y, rand_y)
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)

    acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x,
                                             y_target: rand_y,
                                             prediction_grid:rand_x})
    batch_accuracy.append(acc_temp) 
    if (i+1)%50==0:
        print(sess.run(b_vec_cross, feed_dict={x_data: rand_x, y_target: rand_y, prediction_grid:rand_x}))
        print('b_vec_cross')
        print(sess.run(y_target_cross, feed_dict={x_data: rand_x, y_target: rand_y, prediction_grid:rand_x}))
        print('y_target_cross')
        print(sess.run(test1, feed_dict={x_data: rand_x, y_target: rand_y, prediction_grid:rand_x}))
        print('test1')
# 创建数据点的预测网格，运行预测函数
# print('hehe', val_x[:, 0], np.array([1.0, 1.0, 3]))
# print(np.array([1.0, 1.0, 3]).min())
x_min, x_max = val_x[:, 0].min() - 1, val_x[:, 0].max() + 1
y_min, y_max = val_x[:, 1].min() - 1, val_x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_predictions = sess.run(prediction, feed_dict={x_data: rand_x,
                                                   y_target: rand_y,
                                                   prediction_grid: grid_points})
grid_predictions = grid_predictions.reshape(xx.shape)

# Plot points and grid
plt.contourf(xx, yy, grid_predictions, cmap=plt.cm.Paired, alpha=0.8)
plt.plot(class1_x, class1_y, 'ro', label='I. setosa')
plt.plot(class2_x, class2_y, 'kx', label='I. versicolor')
plt.plot(class3_x, class3_y, 'gv', label='I. virginica')
plt.title('Gaussian SVM Results on Iris Data')
plt.xlabel('Pedal Length')
plt.ylabel('Sepal Width')
plt.legend(loc='lower right')
plt.ylim([-0.5, 3.0])
plt.xlim([3.5, 8.5])
plt.show()

# Plot batch accuracy
plt.plot(batch_accuracy, 'k-', label='Accuracy')
plt.title('Batch Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()