# _*_ coding:utf-8 _*_
"""
    create on 2018-11-15
    create by zhangpeng
"""

import matplotlib.pyplot as plt
import numpy as np

ATT_LSTM = [0.8892, 0.861, 0.9243]
MATT_CNN = [0.8966, 0.8556, 0.9316]
ATT_RLSTM = [0.8867, 0.8543, 0.9344]
CNN_RLSTM = [0.9016, 0.8636, 0.9435]
x = np.arange(3)
total_width, n = 0.8, 4
width = total_width / n
x = x - (total_width- width) / 2
plt.bar(x, ATT_LSTM, color='r', width=width, label='ATT_LSTM')
plt.bar(x + width, MATT_CNN, color='y', width=width, label='MATT_CNN')
plt.bar(x + 2*width, ATT_RLSTM, color='c', width=width, label='ATT_RLSTM')
plt.bar(x + 3*width, CNN_RLSTM, color='g', width=width, label='CNN_RLSTM')
plt.xlabel('dateset')
plt.ylabel('accuracy')
# plt.legend(loc='best')
plt.xticks([0, 1, 2],['REST', 'LAPT', 'AUTO'])
my_y_ticks = np.arange(0.8, 0.95, 0.02)
plt.ylim((0.8, 0.95))
plt.yticks(my_y_ticks)
plt.show()
