# _*_coding: utf-8 _*_
"""
    时间：2018年12月6日
    作者：张鹏
    文件命：utils.py
    功能：数据加载，特征处理，图像显示
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# import cv2
# from cv2 import *
from sklearn.preprocessing import StandardScaler
import time
from sklearn.model_selection import GridSearchCV
from PythonAI.ImageData.FashionMnist import config

def load_fashion_mnist_data(datafile):
    """
        根据给定的fashion_mnist数据集文件读取数据
        参数
            :param datafile: 数据集集
            :return:
                    X： 数据矩阵[n_sample, img_rows, img_cols]
                    y:  标签
    """
    data_df = pd.read_csv(datafile)
    print(data_df)
    X = data_df.iloc[:, 1:].values.astype(np.uint8)
    y = data_df.iloc[:, 0].values.astype(np.uint8)
    print('共有{}个图像'.format(X.shape[0]))
    return X, y