"""
    内容：科学计算库Numpy
    时间：2018年6月28日
"""
import numpy as np
import pandas as pd
# 1 创建Array
my_list = [1, 2, 3, 5]
x = np.array(my_list)
y = pd.Series(my_list)

print(my_list)
print(x)
print(y)

print(np.array([4, 5, 6]) - np.array([1, 2, 3]))
print(pd.Series([4, 5, 6]) - pd.Series([1, 2, 3]))
print(x.ndim)
print(y.ndim)
print(x.shape)
print(y.shape)

n = np.arange(0, 30, 2)
print(n)
print(n.reshape(3, 5))

print(np.ones((3,2)))
print(np.zeros((3,2)))
print(np.eye(3))
print(np.diag(my_list))
print(np.array([1, 2, 3] * 3 ))

# 2 Array操作
p1 = np.arange(1,18,2).reshape((3,3))
p2 = np.ones((3,3))
print(p1 + p2)
print(p1 * p2)
print(p1 ** 2)
print('p1\n',p1)
print('p2\n',p2)
print(p1.dot(p2)) # 两个矩阵相乘

p3 = np.arange(6).reshape(3, 2)
print('p3\n',p3)
print(p3.T)

print(type(p3))
print(p3.dtype)
p4 = p3.astype('float')
print(type(p4))
print(p4.dtype)

p5 = np.array([4,2,4,71,6,13,9,4])
print(p5.sum())
print(p5.std())
print(p5.min())
print(p5.max())
print(p5.argmin())
print(p5.argmax())

# 索引与切片
# 一维
s = np.arange(13)
print(s[:])
print(s[0])
print(s[3:7])
print(s[3:])
print(s[[2, 6]])

# 二维
r = np.arange(36).reshape(6, 6)
print(np.ndim(r))
print(np.shape(r))
print(r)
print(r[2,4])
print(r[2:4,2])
print(r[2,2:4])
print(r > 30)
print(r[r>30])
r[r>30]=30
print(r)

r2 = r[:2, :2]
r2[:] = 0
print(r2)
print(r)
r3 = r.copy() # 操作时不改变原始数据
r3[:] = 0
print(r)
print(r3)

# 遍历Array
t = np.random.randint(0, 10, (4, 3))
print(t)

for row in t:
    print(row)

# 使用enumerate()
for i, row in enumerate(t):
    print('row{}is{}'.format(i, row))

# 使用zip对两个array进行遍历计算
print(t)
t2 = t **2
print(t2)
print('----\n',list(zip(t, t2)))
for i, j in zip(t, t2):
    print('{}+{}={}'.format(i, j, i+j))