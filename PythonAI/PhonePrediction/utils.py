"""
    时间：2018年11月23日 14：49
    作者：张鹏
    文件命：utils.py
    功能：工具类
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from PythonAI_Ⅱ.PhonePrediction import config
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import time


def clearn_data(dataset):

    """
        数据清洗
        参数：
            - dataset: 数据集
        返回值：
            - cln_data: 清晰后的数据
    """

    cln_data = dataset.fillna('moderate')
    return cln_data

def inspect_date(train_data, test_data):
    """
        查看数据集

        参数：
            - train_data:   训练集
            - test_data:    测试集
    """
    print('\n===================== 数据查看 =====================')
    print('训练集数据有{}条'.format(len(train_data)))
    print('测试集数据有{}条'.format(len(test_data)))

    # 可视化各类别数量统计图
    plt.figure(figsize=(10, 5))

    # 训练集数据
    ax1 = plt.subplot(1, 2, 1)
    sns.countplot(x='Risk', data=train_data)

    plt.title('Train_data')
    plt.xlabel('Risk')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2, sharey=ax1)
    sns.countplot(x='Risk', data=test_data)
    plt.title('Test_data')
    plt.xlabel('Risk')
    plt.ylabel('Count')

    plt.tight_layout()
    # plt.show()


def transform_data(data_df):

    """
        特征转换
        参数：
            -data_df: DataFrame数据
        返回：
            -X：转换后的数据特征
            -y：转换后的标签
    """
    trans_data_df = data_df.copy()

    trans_data_df['Sex'] = data_df['Sex'].map(config.sex_dict)
    trans_data_df['Housing'] = data_df['Housing'].map(config.housing_dict)
    trans_data_df['Saving accounts'] = data_df['Saving accounts'].map(config.saving_dict)
    trans_data_df['Checking account'] = data_df['Checking account'].map(config.checking_dict)
    trans_data_df['Purpose'] = data_df['Purpose'].map(config.purpose_dict)
    trans_data_df['Risk'] = data_df['Risk'].map(config.risk_dict)
    print('开始转换')
    X = trans_data_df[config.feat_cols].values
    y = trans_data_df[config.label_col].values

    return X, y


def train_test_model(X_trian, y_train, X_test, y_test, param_range, model_name):
    """
        训练并测试数据

        :param X_trian: 训练数据
        :param y_train: 训练数据
        :param X_test:  测试数据
        :param y_test:  测试数据
        :param param_range: 参数
        :param model_name:   knn, kNN模型，对应参数为 n_neighbors
                             lr, 逻辑回归模型，对应参数为 C
                             dt, 决策树模型，对应参数为 max_depth
                             svm, 支持向量机，对应参数为 C
        :return:
                    1. 最优模型
                    2. 平均训练耗时
                    3. 准确率
    """
    models = []
    durations = []
    scores = []
    for param in param_range:
        if model_name == 'KNN':
            print('正在训练KNN模型，当前参数K={}'.format(param), end='')
            model =KNeighborsClassifier(n_neighbors=param)
        elif model_name == 'LR':
            print('正在训练Logistic Regression模型，当前参数C={}'.format(param), end='')
            model = LogisticRegression(C=param)
        elif model_name == 'DT':
            print('正在训练决策树模型，当前参数max_depth={}'.format(param), end='')
            model = DecisionTreeClassifier(max_depth=param)
        elif model_name == 'SVM':
            print('正在训练决策树模型,当前参数C={}'.format(param), end='')
            model = SVC(C=param,kernel='linear')
        else:
            print('请输入正确的模型')

        start_time = time.time()
        # 训练模型
        model.fit(X_trian, y_train)
        # 计时
        end_time = time.time()
        duration = end_time - start_time
        print('\n参数为{}的{}模型耗时{:.4f}s'.format(param, model_name, duration))

        # 模型验证
        score = model.score(X_test, y_test)
        print('参数为{}的{}模型准确率{:.4f}'.format(param, model_name, score ))

        models.append(model)
        durations.append(duration)
        scores.append(score)
    mean_duration = np.mean(durations)
    print('{}模型训练的平局时间为{:.4f}'.format(model_name, mean_duration))

    best_idx = np.argmax(scores)
    best_acc = scores[best_idx]
    best_model = models[best_idx]

    return best_model, best_acc, mean_duration