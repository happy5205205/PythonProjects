# _*_ coding:utf-8 _*_
"""
    时间：2018年11月28日
    作者:张鹏
    文件命：utils.py
    功能：工具文件
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from PythonAI_Ⅱ.AnimalPrediction import config


def inspect_dataset(trian_data, test_data):
    """
        查看数据集
            - trian_data:训练数据
            - test_data:测试数据
    """
    print('\n===================== 数据查看 =====================')
    print('训练数据有{}条'.format(len(trian_data)))
    print('测试数据有{}条'.format(len(test_data)))
    # print(config.label_col[0])

    # 可视化试图
    plt.figure(figsize=(10, 5))

    # 训练数据
    ax1 = plt.subplot(1, 2, 1)
    sns.countplot(x=config.label_col[0], data=trian_data)
    plt.title('Training set')
    plt.xlabel('Class')
    plt.ylabel('Count')

    # 测试数据
    plt.subplot(1, 2, 2, sharey=ax1)
    sns.countplot(x=config.label_col[0], data=test_data)
    plt.title('Testing set')
    plt.xlabel('Class')
    plt.ylabel('Count')

    plt.tight_layout()
    # plt.show()


def transform_data(train_data, test_data):
    """
        将类别型特征进行独热编码
        数字型特征进行0-1归一化
        使用PCA降维
            - train_data:  训练数据
            - test_data:   测试数据
            - X_train:    训练数据处理后的特征
            - X_test:     测试数据处理后的特征
    """
    # 独热编码处理类别型特征
    encoder = OneHotEncoder(sparse=False)
    X_train_cat_feat = encoder.fit_transform(train_data[config.category_cols].values)
    X_test_cat_feat = encoder.transform(test_data[config.category_cols].values)

    # 范围归一化处理数值型特征
    scaler = MinMaxScaler()
    X_train_num_feat = scaler.fit_transform(train_data[config.num_cols].values)
    X_test_num_feat = scaler.transform(test_data[config.num_cols].values)

    # 合并所有特征
    X_train_raw = np.hstack((X_train_cat_feat, X_train_num_feat))
    X_test_raw = np.hstack((X_test_cat_feat, X_test_num_feat))

    print('特征处理后，特征维度为: {}（其中类别型特征维度为: {}，数值型特征维度为: {}）'.format(
        X_train_raw.shape[1], X_train_cat_feat.shape[1], X_train_num_feat.shape[1]))

    # 使用特征降维
    pca = PCA(n_components=0.99)
    X_train = pca.fit_transform(X_train_raw)
    X_test = pca.transform(X_test_raw)

    print('PCA特征降维后，特征维度为: {}'.format(X_train.shape[1]))

    return X_train, X_test

def train_test_model(X_train, y_train, X_test, y_test, model_name, model, param_range):
    """
        测试并训练模型
            model_name:
            kNN         kNN模型，对应参数为 n_neighbors
            LR          逻辑回归模型，对应参数为 C
            SVM         支持向量机，对应参数为 C
            DT          决策树，对应参数为 max_depth
            Stacking    将kNN, SVM, DT集成的Stacking模型， meta分类器为LR
            AdaBoost    AdaBoost模型，对应参数为 n_estimators
            GBDT        GBDT模型，对应参数为 learning_rate
            RF          随机森林模型，对应参数为 n_estimators

        根据给定的参数训练模型，并返回
        1. 最优模型
        2. 平均训练耗时
        3. 准确率
    """
    print('训练{}......'.format(model_name))

    clf = GridSearchCV(estimator=model, param_grid=param_range, cv=5,
                       scoring='accuracy', refit=True)
    start_time = time.time()
    clf.fit(X_train, y_train)
    # 计时
    end_time = time.time()
    duration = end_time - start_time
    print('耗时{:.4f}s'.format(duration))

    # 验证模型
    train_score = clf.score(X_train, y_train)
    print('训练准确率{:.3f}%'.format(train_score * 100))
    test_score = clf.score(X_test, y_test)
    print('测试准确率{:.3f}%'.format(test_score))

    return clf, test_score, duration