import numpy as np
print('找出1000内的偶数(for循环)：')
l1 = []
for i in range(11):
    if i % 2 == 0:
        l1.append(i)
print(np.array(l1).reshape(3,2))#转换乘3行2列数组


# 列表推导式,找出1000以内的偶数

l2 = [i for i in range(1000) if i % 2 == 0]
print(l2)