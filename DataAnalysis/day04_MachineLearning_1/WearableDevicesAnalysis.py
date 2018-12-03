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
    版本：jupyter版本
'''
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']   # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像是负号‘-’显示为方块的问题

print('*******************************指定数据文件*******************************')
# 指定数据文件
data_path = './data'
train_datafile = os.path.join(data_path, 'train.csv')
test_datafile = os.path.join(data_path, 'test.csv')
print('====指定数据文件成功====')

print('*******************************加载数据文件*******************************')
# 加载数据文件
train_data = pd.read_csv(train_datafile)
test_data = pd.read_csv(test_datafile)
print('====数据加载成功====')

print('*******************************查看数据详情*******************************')
# 查看数据详情
print('训练集的数据量为：{}'.format(len(train_data)))
print('训练集的数据详情', train_data.info())
print('测试集的数据量为：{}'.format(len(test_data)))
print('测试集的数据详情', test_data.info())

print('*******************************可视化统计*******************************')
# 可视化各类别的数据量统计
plt.figure(figsize=(12, 6))
ax1 = plt.subplot(1,2,1)
sns.countplot(x='Activity', data=train_data)
plt.title('训练集')
plt.xticks(rotation= 'vertical')
plt.xlabel('行为类别')
plt.ylabel('数量')
# plt.show()

plt.subplot(1, 2, 2, sharey = ax1)
sns.countplot(x='Activity', data=test_data)
plt.title('测试集')
plt.xticks(rotation= 'vertical')
plt.xlabel('行为类别')
plt.ylabel('数量')
plt.tight_layout()
# plt.show()

print('*******************************构建训练测试数据*******************************')
# 构建训练测试数据
print('*******************************特征处理*******************************')
# 特征处理
feat_name = train_data.columns[: -2].tolist()
# print(feat_name)
# print(len(feat_name))
X_train = train_data[feat_name].values
# print(X_train)
X_test = test_data[feat_name].values
# print(X_test)
print('共有{}维特征'.format(X_train.shape[1]))

print('*******************************标签处理*******************************')
# 标签处理
train_labels = train_data['Activity'].values
# print(train_labels)
test_labels = test_data['Activity'].values
# print(test_labels)

# 使用sklearn.preprocessing中的LabelEncoder进行类别标签处理
from sklearn.preprocessing import LabelEncoder
label_enc = LabelEncoder()
y_train = label_enc.fit_transform(train_labels)
# print(y_train)
y_test = label_enc.transform(test_labels)
# print(y_test)

# 输出类别标签
print('类别标签：', label_enc.classes_)
for i in range(len(label_enc.classes_)):
    print('编码{}对应标签为:{}'.format(i, label_enc.inverse_transform(i)))

print('*******************************数据建模及验证*******************************')
# 数据建模及验证
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

print('*******************************KNN模型*******************************')
k_range = [5, 10, 15]
knn_model = []
knn_score = []
knn_durations = []

for k in k_range:
    print('训练KNN模型，当K={}.....'.format(k), end='')
    # 创建模型
    knn = KNeighborsClassifier(n_neighbors=k)
    # 训练模型
    start = time.time()
    knn.fit(X_train, y_train)

    # 计时
    end = time.time()
    duration = end - start
    print('耗时{:.4f}s'.format(duration), end=',')

    # 验证模型
    score = knn.score(X_test, y_test)
    print('准确率为：{:.3f}'.format(score))

    knn_model.append(knn)
    knn_durations.append(duration)
    knn_score.append(score)

# print('knn_model:',knn_model)
knn_mean_duration = np.mean(knn_durations)
print('训练KNN平均耗时:{}s'.format(knn_mean_duration))
# 记录最优模型
# print('knn_score',knn_score)
best_idx = np.argmax(knn_score)
# print('best_idx=', best_idx)
best_knn_acc = knn_score[best_idx]
# print('best_knn_acc=', best_knn_acc)
print('最优模型的KNN，K={}, 准确率为：{:.3f}'.format(knn_model[best_idx].get_params()
                                          ['n_neighbors'],best_knn_acc))

print('*******************************逻辑回归模型*******************************')
lr_range = [0.1, 1, 100]
lr_models = []
lr_scores = []
lr_durations = []
for c in lr_range:
    print('训练逻辑回归模型，当C={}.....'.format(c),end='')
    #创建模型
    lr_model = LogisticRegression(C=c)
    #训练模型
    start = time.time()
    lr_model.fit(X_train, y_train)

    # 计时
    end = time.time()
    duration = end - start
    print('，耗时{:.4f}s，'.format(duration), end='')

    # 验证模型
    score = lr_model.score(X_test, y_test)
    print('准确率：{:.3f}'.format(score))

    lr_models.append(lr_model)
    lr_durations.append(duration)
    lr_scores.append(score)

lr_mean_duration = np.mean(lr_durations)
print('逻辑回归模型的平均时长为{:.4f}s'.format(lr_mean_duration))

# 记录最优模型
best_idx = np.argmax(lr_scores)
best_lr_acc = lr_scores[best_idx]
print('最优的逻辑回归模型，当C={}时，准确率为：{:.3f}'.format(lr_models[best_idx].get_params()['C'],
                                      best_lr_acc))

print('*******************************SVM模型*******************************')
c_range = [100, 1000, 10000]
svm_models = []
svm_scores = []
svm_durations = []
for c in c_range:
    print('训练SVM模型，当C={}时'.format(c), end=',')
    # 创建模型
    svm_model = SVC(C= c)

    # 训练模型
    start = time.time()
    svm_model.fit(X_train, y_train)

    # 计时
    end = time.time()
    duration = end - start
    print('耗时{:.4f}s'.format(duration), end=',')

    # 验证模型
    score = svm_model.score(X_test, y_test)
    print('准确率为{:.3f}'.format(score))

    svm_models.append(svm_model)
    svm_durations.append(duration)
    svm_scores.append(score)

svm_mean_duration = np.mean(svm_durations)
print('训练SVM模型的平均时长为：{:.4f}s'.format(svm_mean_duration))

# 记录最优模型
best_idx = np.argmax(svm_scores)
best_svm_acc = svm_scores[best_idx]
print('最优的SVM模型， 当C={}时，准确率为：{:.3f}'.format(svm_models[best_idx].get_params()['C'],
                                        best_svm_acc))

print('*******************************决策树模型*******************************')
depth_range = [50, 100, 150]
tree_models = []
tree_scores = []
tree_durations = []
for depth in depth_range:
    print('训练决策树，当max_depth={}时'.format(depth), end=',')
    # 创建模型
    tree_model = DecisionTreeClassifier(max_depth=depth)

    # 训练模型
    start = time.time()
    tree_model.fit(X_train, y_train)

    # 计时
    end = time.time()
    duration = end - start
    print('耗时{:.4f}s'.format(duration), end=',')

    # 验证模型
    score = tree_model.score(X_test, y_test)
    print('准确率为：{:.3f}'.format(score))

    tree_models.append(tree_model)
    tree_durations.append(duration)
    tree_scores.append(score)

tree_mean_duration = np.mean(tree_durations)
print('训练决策树的平均时长为：{:.4f}s'.format(tree_mean_duration))

# 记录最优模型
best_idx = np.argmax(tree_scores)
best_tree_acc = tree_scores[best_idx]
print('最优的决策树模型，max_depth={}, 准确率为：{:.3f}'.format(tree_models[best_idx].get_params()
                                                  ['max_depth'],best_tree_acc))

print('*******************************模型及结果比较*******************************')
results_df = pd.DataFrame(columns= ['Accuracy(%)', 'Times'], index=['kNN', 'LR', 'SVM', 'DT'])
results_df['Accuracy(%)'] = [best_knn_acc * 100, best_lr_acc * 100,
                             best_svm_acc * 100, best_tree_acc * 100]
results_df['Times'] = [knn_mean_duration, lr_mean_duration, svm_mean_duration, tree_mean_duration]
# print(results_df)
plt.figure(figsize=(12, 6))
ax1 = plt.subplot(1, 2, 1)
results_df.plot(y=['Accuracy(%)'], kind='bar', ylim=[80, 100], ax=ax1, title='准确率（%）', legend=False)
ax2 = plt.subplot(1, 2, 2)
results_df.plot(y=['Times'], kind='bar', ax=ax2, title='训练耗时', legend= False)
plt.savefig('./各类模型对比')
plt.show()