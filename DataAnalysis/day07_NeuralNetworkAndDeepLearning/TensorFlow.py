# -*- coding: utf-8 -*-
"""
    时间：2018年4月24日
    内容：TensorFlow基本学习
    作者：张鹏
"""
import tensorflow as tf
# 1 计算模式：计算图

a = tf.constant([1, 2], name='a')
b = tf.constant([1, 2], name='b')

result = a + b

# print('result\n', result)
# 获取a结果的维度
print('tf.Session().run(a)\n', tf.Session().run(a))

# 若果没有特殊声明，a.graph =返回其所属的计算图，即默认计算图
print(a.graph == tf.get_default_graph())

# 2 数据模式：张量

a1 = tf.constant([1, 2], name='a1')
b1 = tf.constant([1,2], name='b1')
result1 = tf.add(a1, b1, name='add')
# print('type(a1)', type(a1))
# print('result1\n', result1)
print('tf.Session().run(result1)\n', tf.Session().run(result1))

# 3 运行模式：会话
# 创建一个会话
sess =tf.Session()
print(sess.run(result))
# 关闭会话
# sess.close()

# 通过python上下文管理器来管理对话
with sess.as_default():
    print(sess.run(result))

print(sess.run(result))
print(result.eval(session=sess))

#
tf.InteractiveSession()
print(result.eval())
# sess.close()

# 4 Placeholder
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # provides a shortcut for tf.add(a, b)

print(sess.run(adder_node, {a: 3, b: 5}))
print(sess.run(adder_node, {a: [1, 2], b: [3, 4]}))

add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a: 3, b: 1}))

# Variable
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype= tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model, {x: [1, 2, 3,4]}))

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

fixW = tf.assign(W, [-.1])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print('11111\n', sess.run(loss, {x: [1, 2, 3, 4], y: [0 , -1, -2, -3]}))

# tf.train
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init) # 重新设置初始化参数
for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W, b]))
