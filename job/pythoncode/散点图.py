# _*_ coding: utf-8 _*_

"""
    create on 2018-11-15
    create by zhangpeng
"""
import matplotlib.pyplot as plt
from sklearn import datasets

X_train, y_train = datasets.load_breast_cancer(return_X_y=True) # 加载数据集
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', s=40,
            cmap=plt.cm.Spectral)   # c--color，s--size,marker点的形状
plt.show()