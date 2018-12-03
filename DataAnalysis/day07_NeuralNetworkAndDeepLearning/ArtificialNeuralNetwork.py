# -*- coding: utf-8 -*-
"""
    时间：2018年4月23日
    作者：张鹏
    内容：人工神经网络
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# 加载数据
fruits_df = pd.read_table('./fruit_data_with_colors.txt')
print(fruits_df.head(3))

# 数据处理:特征和标签处理
X = fruits_df[['mass', 'width', 'height']]
y = fruits_df['fruit_label'].copy() # 处理时不会改变原始数据

# 标签处理，将不是apple标签的全部设置为0
y[y != 1] = 0

# 数据集的划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/4, random_state=0)

# ANN建模
# 1 单层ANN
from sklearn.neural_network import MLPClassifier
print('\n===================== 单层ANN =====================')
# 神经元个数
units = [1, 10, 100]
for unit in units:
    # 激活函数： relu,logistic, tanh
    ann_model = MLPClassifier(hidden_layer_sizes=[unit], activation='relu',
                              solver='lbfgs', random_state=0)
    ann_model.fit(X_train, y_train)
    print('当神经元个数为{}时，准确率为{:.3f}'.format(unit, ann_model.score(X_test, y_test)))
print('\n===================== 多层神经元 =====================')
# 2 多层神经元
ann_model = MLPClassifier(hidden_layer_sizes=[10, 10], activation='relu',
                          solver='lbfgs', random_state=0)
ann_model.fit(X_train, y_train)
print('当神经元个数为{}时，准确率为{:.3f}'.format(unit, ann_model.score(X_test, y_test)))

# ANN中的正则化
# aplhas
print('\n===================== ANN中的正则化 =====================')
aplhas = [0.001, 0.01, 0.1, 1]
for aplha in aplhas:
    ann_model = MLPClassifier(hidden_layer_sizes=[100, 100], activation='relu',
                              solver='lbfgs', random_state=0, alpha=aplha)
    ann_model.fit(X_train, y_train)
    print('当神经元个数为{}时，准确率为{:.3f}'.format(unit, ann_model.score(X_test, y_test)))

