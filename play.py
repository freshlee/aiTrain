from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf;
import numpy as np;
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# sess = tf.InteractiveSession()

# x = tf.placeholder("float", shape=[None, 784])
# y_ = tf.placeholder("float", [None,10])

# W = tf.Variable(tf.zeros([784,10]))
# b = tf.Variable(tf.zeros([10]))

# with tf.name_scope('graph') as scope:
#     y__ = tf.matmul(x,W) + b
#     y = tf.nn.softmax(y__)


# cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# writer = tf.summary.FileWriter("logs/", sess.graph)
# init = tf.initialize_all_variables()
# sess = tf.Session()
# sess.run(init)

# for i in range(1000):
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
# import os
# os.system('tensorboard --logdir=G:\人工智能学习\logs')
# print(sess.run(y, feed_dict={x: batch_xs, y_: batch_ys}))
# print(batch_ys)

# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
tf.reset_default_graph()
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=10) # state_size = 128
print(cell.state_size) # 128

batch_size = 32
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# input_data = np.random.rand(32,100) 
# print(xs[0], len(ys))

with tf.Session() as sess:
    inputs = tf.placeholder(np.float32, shape=(batch_size, 784))
    y_ = tf.placeholder(np.float32, shape=(batch_size, 10))
    h0 = cell.zero_state(batch_size, np.float32) # 通过zero_state得到一个全0的初始状态，形状为(batch_size, state_size)
    output, h1 = cell.call(inputs, h0) #调用call函数
    # print(h0.shape, output.shape) # (32, 128)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    init = tf.global_variables_initializer()
    sess.run(init)
    # 训练
    for i in range(1000):
        xs, ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={inputs: xs, y_: ys})
        if i % 25 == 0:
            output_now = sess.run(output, feed_dict={inputs: xs, y_: ys})
            print('round')
            print(np.argmax(ys, 1))
            print(np.argmax(output_now, 1))