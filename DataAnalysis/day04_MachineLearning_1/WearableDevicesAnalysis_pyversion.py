'''
    时间：2018-3-8
    作者：张鹏
    项目名称：实战案例4-1：根据可穿戴设备识别用户行为
    项目描述：
            用户行为识别数据集是通过采集30天用户的行为创建的。数据是由绑定在用户腰间的智能手机记录的，该智能手机内嵌有传感器。
        创建该数据集的目的是用于识别/分类6组不同的用户行为。
            数据集中的用户是由19-48岁间的30个志愿者组成的。戴有智能手机（Samsung Galaxy S II）的每个志愿者会做出6个行
        为（WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING）。
        通过智能手机的加速计和陀螺仪能够以50Hz的频率采集3个方向的加速度和3个方向的角速度。采集后的数据集随机分为两部分，
        70%用于模型的训练，30%用于模型的验证。
            传感器信号已经预处理去除了噪声，并且在固定时间窗口（2.56s）内进行采样，
        每两个窗口间有50%的重叠部分（每个窗口有128个数据）。每个时间窗口同时提供时间和频率上的统计数据作为特征。
    数据集描述:
        Kaggle提供的数据集。数据集包含训练集（train.csv）和测试集（test.csv），形式均为CSV文件。
        每条记录提供有以下数据
        3个方向的加速度，估计的身体加速度，3个方向的角速度。最终是561维的向量。
        对应的标签
        志愿者编号
    任务：1 数据查看
         2 数据建模及验证
         3 模型及结果比较
    版本：python版本
'''

import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] =False


def train_model(X_train, y_train, X_test, y_test, model_name, model_params):
    '''
        不同模型间的训练
        knn, kNN模型，对应参数为 n_neighbors
        lr, 逻辑回归模型，对应参数为 C
        svm, SVM模型，对应参数为 C
        dt, 决策树模型，对应参数为 max_dpeth
    '''
    models = []
    durations = []
    scores = []
    for params in model_params:
        # 创建模型
        if model_name == 'KNN':
            print('训练KNN模型，当K={}时'.format(params), end=',')
            model = KNeighborsClassifier(n_neighbors=params)
        elif model_name == 'LR':
            print('训练逻辑回归模型，当C={}时'.format(params), end=',')
            model = LogisticRegression(C=params)
        elif model_name == 'SVM':
            print('训练SVM模型，当C={}时'.format(params), end=',')
            model = SVC(C=params)
        else:
            model_name == 'DT'
            print('训练决策树模型，当max_depth={}'.format(params), end=',')
            model = DecisionTreeClassifier(max_depth=params)

        # 计时
        start = time.time() # 开始时间
        # 训练模型
        model.fit(X_train, y_train)
        end = time.time() # 结束时间
        duration = end - start
        print('耗时{:.4f}s'.format(duration), end=',')

        # 验证模型
        score = model.score(X_test, y_test)
        print('准确率为：{:.3f}'.format(score))

        models.append(model)
        scores.append(score)
        durations.append(duration)

    mean_duration = np.mean(durations)
    print('{}平均耗时为{:.4f}s'.format(model_name, mean_duration))

    # 记录最有模型
    best_idx = np.argmax(scores)
    best_acc = scores[best_idx]
    best_model = models[best_idx]
    return best_model, best_acc, mean_duration


def main():
    # 指定数据路径
    data_path = './data'
    train_datafile = os.path.join(data_path, 'train.csv')
    test_datafile = os.path.join(data_path, 'test.csv')
    # 加载数据文件
    train_data= pd.read_csv(train_datafile)
    test_data = pd.read_csv(test_datafile)
    # 任务1. 数据查看
    print('\n===================== 任务1. 数据查看 =====================')
    print('训练集有{}条记录'.format(len(train_data)))
    print('测试集有{}条记录'.format(len(test_data)))
    # 可视化个类别数据量统计
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(1, 2, 1)
    plt.title('训练集')
    sns.countplot(x='Activity', data=train_data)
    plt.xlabel('行为类别')
    plt.xticks(rotation='vertical')
    plt.ylabel('数量')
    # plt.show()
    plt.subplot(1, 2, 2, sharey=ax1)
    plt.title('测试集')
    sns.countplot(x='Activity', data=test_data)
    plt.xticks(rotation = 'vertical')
    plt.xlabel('行为类别')
    plt.ylabel('数量')
    # plt.show()

    # 特征处理
    feature_name = train_data.columns[:-2].tolist()
    # print(feature_name)
    print('共有{}个特征'.format(len(feature_name)))
    X_train = train_data[feature_name].values
    X_test = test_data[feature_name].values

    # 标签处理
    train_label = train_data['Activity'].values
    test_label = test_data['Activity'].values

    # 使用sklearn.preprocessing.LabelEncoder进行类别标签处理
    from sklearn.preprocessing import LabelEncoder
    label_enc = LabelEncoder()
    y_train = label_enc.fit_transform(train_label)
    y_test = label_enc.transform(test_label)

    print('类别标签：{}'.format(label_enc.classes_))

    for i in range(len(label_enc.classes_)):
        print('编码{}对应的标签为{}'.format(i, label_enc.inverse_transform(i)))

     # 任务2. 数据建模及验证
    print('\n===================== 任务2. 数据建模及验证 =====================')
    model_name_param_dict = {'KNN':[5, 10, 15], 'LR':[0.01, 1, 100],
                             'SVM':[100, 1000, 10000], 'DT':[50, 100, 150]}
    result_df = pd.DataFrame(columns=['Activity', 'time'], index=list(model_name_param_dict.keys()))
    for model_name, model_param in model_name_param_dict.items():
        _,best_acc,duration_mean = train_model(X_train, y_train, X_test, y_test, model_name, model_param)

        result_df.loc[model_name, 'Activity'] = best_acc * 100
        result_df.loc[model_name, 'time'] = duration_mean
    # 任务3. 模型及结果比较
    print('\n===================== 任务3. 模型及结果比较 =====================')
    plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(1, 2, 1)
    result_df.plot(y=['Activity'], kind='bar', ylim=[80, 100], ax=ax1, title='准确率(%)', legend=False)

    ax2 = plt.subplot(1, 2, 2)
    result_df.plot(y=['time'], kind='bar', ax=ax2, title='训练耗时(s)', legend=False)
    plt.savefig('./pred_results.png')
    plt.show()

if __name__ == '__main__':
    main()