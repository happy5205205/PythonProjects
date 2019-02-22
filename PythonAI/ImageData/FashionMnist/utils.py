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
import cv2
from sklearn.preprocessing import StandardScaler
import time
from sklearn.model_selection import GridSearchCV
from PythonAI.ImageData.FashionMnist import config, utils


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
    # print(data_df.iloc[2, :])
    X = data_df.iloc[:, 1:].values.astype(np.uint8)
    y = data_df.iloc[:, 0].values.astype(np.uint8)
    print('共有{}个图像'.format(X.shape[0]))
    return X, y


def polt_random_samples(X):
    """
        随机选取9张图像数据进行可视化
        参数：
            :param X: 数据矩阵[n_samples, img_rows * img_cols]
    """
    random_X = X[np.random.choice(X.shape[0], 9, replace=False), :]
    for i in range(9):
        img_data = random_X[i, :].reshape(config.img_rows, config.img_cols)
        plt.subplot(3, 3, i+1)
        plt.imshow(img_data, cmap='gray')
        plt.tight_layout()
    # plt.show()


def extract_feats(X):
    """
        特征提取：
        参数：
            :param X: 数据矩阵[n_samples, img_rows * img_cols]
            :return: -feat_arr: 特征矩阵
    """
    n_sample = X.shape[0]
    feat_list = []
    for i in range(n_sample):
        img_data = X[i, :].reshape(config.img_rows, config.img_cols)
        # 中值滤波， 去除噪声
        blur_img_data = cv2.medianBlur(img_data, 3)

        # 直方图均衡化
        equ_blur_img_data = cv2.equalizeHist(blur_img_data)

        # 将图像转换成特征向量
        feat = equ_blur_img_data.flatten()
        feat_list.append(feat)

        if (i + 1) % 5000 ==0:
            print('已完成{}个特征的提取。'.format(i + 1))
    feat_arr = np.array(feat_list)
    return feat_arr

def do_feature_engineering(feats_train, feats_test):
    """
        特征处理
        参数
            :param feats_train: 训练数据特征矩阵
            :param feats_test:  测试数据特征矩阵
        :return:
    """
    std_scaler = StandardScaler()
    scaled_feats_train = std_scaler.fit_transform(feats_train.astype(np.float64))
    scaled_feats_test = std_scaler.transform(feats_test.astype(np.float64))

    return scaled_feats_train, scaled_feats_test


def train_and_test_model(X_train, y_trian, X_test, y_test, model_name, model, param_range):
    """
        根据给定模型并返回
            1.最优模型
            2.平均训练耗时
            3.准确率
            :param X_train:
            :param y_trian:
            :param X_test:
            :param y_test:
            :param model_name:
            :param model:
            :param param_range:
        :return:
    """
    print('训练{}。。。'.format(model_name))
    clf = GridSearchCV(estimator=model, param_grid=param_range,
                       cv=3, scoring='accuracy', refit=True)
    # 计时
    start = time.time()
    clf.fit(X_train, y_trian)
    end = time.time()
    duration = end - start
    print('{}模型耗时{:.4f}s'.format(model_name, duration))

    # 验证模型
    print('训练数据准确率：{:.3f}'.format(clf.score(X_train, y_trian)))

    score = clf.score(X_test, y_test)
    print('测试数据准确率{:.3f}'.format(score))

    return clf, score, duration