# -*- coding = utf-8 -*-
"""
    time:2018-09-26
    create by zhangpeng
"""
import tensorflow as tf
import numpy as np

# 使用 Numpy生成假数据（phony data），总共100个点
x_data = np.float32(np.random(2, 100)) #随机数
y_data = np.dot([0.100, 0.200], x_data) + 0.300

