# -*- coding: utf-8  -*-
"""
    时间：2018年4月24日
    内容：TensorFlow实现卷积神经网络（CNN）
    作者：张鹏
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 加载数据集，并创建默认的InteractiveSession
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

# 创建权重和偏置的初始化函数以便重复使用
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
     # 为偏置增加一些小的正值0.1，避免dead neurons
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 定义卷积层和池化层

def conv2d(x, W):
    # tf.nn.conv2d
    # x: 输入
    # W: 卷积参数，如 [5, 5, 1, 32]表示卷积核的大小为5x5，通道个数为1，卷积核个数为32
    # strides: 移动的步长
    # padding: 边界处理方式
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # tf.nn.max_pool
    # x: 输入的tensor [batch, height, width, channels]
    # ksize: 核的大小，[1, 2, 2, 1]表示在1个样本上，2x2大小的核，1个通道
    # strides：与ksize类似
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 定义placeholder，x为特征，y_为真实的标签
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1]) # reshape为28x28的图像，1个通道，-1表示样本数量不固定

# 定义第一个卷积层
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 定义第二个卷积层
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 经过两次步长为2x2的池化，大小为原始图像的1/4，即7x7
# 定义最后的全连接层
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)

# 为避免过拟合，使用一个Dropout层，训练时随机丢弃一部分节点的数据来减轻过拟合
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 将Dropout层连接最终的Softmax层，得到最终的概率输出
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# 定义损失函数和优化算法
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdadeltaOptimizer(1e-4).minimize(cross_entropy)

# 定义准确率的计算
cross_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(cross_prediction, tf.float32))

# 开始训练过程
tf.global_variables_initializer().run()
for i in range(200):
    batch = mnist.train.next_batch(50)
    if i % 100 ==0:
        train_accuracy = accuracy.eval(feed_dict= {x:batch[0], y_: batch[1],keep_prob:1.0})
        print('step: %d , train_accuracy: %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# 训练完后，在测试集上进行测试
print('test accuracy: %g' % accuracy.eval(feed_dict= {x: mnist.test.images,
                                                      y_: mnist.test.labels,
                                                      keep_prob:1.0}))