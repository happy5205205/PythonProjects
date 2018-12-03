import pandas as pd
import numpy as np
# 1、二项分布
# 1次试验中，成功的次数
# print(np.random.binomial(1, 0.5))
# print()
# 1000次试验中，成功的次数
# print('1000次试验中，成功的次数：', np.random.binomial(1000, 0.5))
# 1000次试验中，成功的概率
# print('1000次试验中，成功的概率：', np.random.binomial(1000, 0.5) / 1000)
# 10000次试验中，成功的次数
# print('10000次试验中，成功的次数：', np.random.binomial(1000, 0.5))
# 10000次试验中，成功的概率
# print('10000次试验中，成功的概率：', np.random.binomial(1000, 0.5) / 1000)

# 模拟100000次试验中，连续两次都是1的次数
total = 10000
events = np.random.binomial(1, 0.5, total)
two_in_1 = 0
for i in range(1, total-1):
    if events[i]== 1 and events[i- 1]== 1:
        two_in_1 += 1
# print('{}次投掷硬币，其中连续两次事1的次数为：{}，概率为{}'.format(total, two_in_1, two_in_1/total))

# 2. 正态分布
# 从正态分布中采样
print(np.random.normal(loc=0.7))

samples = np.random.normal(loc=0.7, size=1000)
print('期望：{}'.format(np.mean(samples)))
print('标准差：{}'.format(np.std(samples)))
