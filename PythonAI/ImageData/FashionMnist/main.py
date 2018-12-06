# _*_ coding: utf-8 _*_
"""
    时间：2018年12月6日
    作者：张鹏
    文件命：main.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from PythonAI.ImageData.FashionMnist import config, utils

IS_SAMPLE_EXP = False


def main():
    """
        主程序
    """
    # 加载数据
    print('加载训练数据。。。。')
    X_train, y = utils.load_fashion_mnist_data(config.train_data_file)
    X_test, y = utils.load_fashion_mnist_data(config.test_data_file)

    print(type(X_test))

if __name__ == '__main__':
    main()