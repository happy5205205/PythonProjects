'''
    时间：2018-4-17
    作者：张鹏
    项目名称：实战案例4-2：根据可穿戴设备识别用户行为
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
import csv
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# 中文乱码处理
plt.rcParams['font.sans-serif'] = ['Simhei']
plt.rcParams['axes.unicode_minus'] = False

# 文件读取
data_path = './data'
train_datafile = os.path.join(data_path, 'train.csv')
test_datafile = os.path.join(data_path, 'test.csv')
train_data = pd.read_csv(train_datafile)
test_data = pd.read_csv(test_datafile)
print('训练数据有{}条记录'.format(len(train_data)))
print('测试数据有{}条记录'.format(len(test_data)))

# 可视化各类别的数量统计
plt.figure(figsize=(10,4))
# 训练集
ax1 = plt.subplot(1, 2, 1)
sns.countplot(x='Activity', data=train_data)

plt.title('Train Data')
plt.xticks(rotation = 'vertical')
plt.xlabel('行为类别')
plt.ylabel('数量')

# 测试集
ax2 = plt.subplot(1, 2, 2)
sns.countplot(x='Activity', data=test_data)
plt.xticks(rotation = 'vertical')
plt.title('Test Data')
plt.xlabel('行为类别')
plt.ylabel('数量')
# plt.show()

# 构建训练测试数据
# 特征处理
feat_names = train_data.columns[:-2].tolist()
#     print(feat_names）
X_train = train_data[feat_names].values
# print(X_train)
X_test = test_data[feat_names].values
print('共有{}个维度'.format(len(feat_names)))

# 标签处理
train_labels = train_data['Activity'].values
test_labels = test_data['Activity'].values

from sklearn.preprocessing import LabelEncoder
label_enc = LabelEncoder()
y_train = label_enc.fit_transform(train_labels)
y_test = label_enc.transform(test_labels)

print('标签类别：\n {}'.format(label_enc.classes_))

for i in range(len(label_enc.classes_)):
    print('编码{}，对应的标签为{}'.format(i, label_enc.classes_[i])) # 我自己新创作
    # print('编码 {} 对应标签 {}。'.format(i, label_enc.inverse_transform(i)))# 老师写的

# 特征处理
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1 max min 标准化
max_min_scaler = MinMaxScaler()
X_train_max_min_scaler = max_min_scaler.fit_transform(X_train)
X_test_max_min_scaler = max_min_scaler.transform(X_test)

# 2 Standard Scaler
std_scaler = StandardScaler()
X_train_std_scaler = std_scaler.fit_transform(X_train)
X_test_std_scaler = std_scaler.transform(X_test)

# 模型建设及参数调整
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# 交叉验证几折
# 如果为了节省训练时间，可以将cv设较小的值，如 3
# 如果为了能找到模型的最优参数，可以选择较大的cv值，如 10
cv_val = 3

# 1 knn
print('************************KNN*******************************')
parameters = {'n_neighbors':[10, 50, 100]}
clf = GridSearchCV(KNeighborsClassifier(), parameters, cv=cv_val, scoring='accuracy')

# max min scaler
clf.fit(X_train_max_min_scaler, y_train)
print('Max Min归一化特征：')
print('最优参数：{}'.format(clf.best_params_))
print('验证集最高得分：{:.3f}'.format(clf.best_score_))
print('测试集准确率：{:.3f}'.format(clf.score(X_test_max_min_scaler, y_test)))

# std 标准化结果
clf.fit(X_train_std_scaler, y_train)
print('std归一化特征：')
print('最优参数：{}'.format(clf.best_params_))
print('验证集最高得分：{:.3f}'.format(clf.best_score_))
print('测试集准确率：{:.3f}'.format(clf.score(X_test_std_scaler, y_test)))

# 2 逻辑回归
print('************************逻辑回归*******************************')
parameters = {'C':[0.1, 1, 10]}
clf = GridSearchCV(LogisticRegression(), parameters, cv=cv_val, scoring='accuracy')

# max min scaler
clf.fit(X_train_max_min_scaler, y_train)
print('Max Min归一化特征：')
print('最优参数：{}'.format(clf.best_params_))
print('验证集最高得分：{:.3f}'.format(clf.best_score_))
print('测试集准确率：{:.3f}'.format(clf.score(X_test_max_min_scaler, y_test)))

# std 标准化结果
clf.fit(X_train_std_scaler, y_train)
print('std归一化特征：')
print('最优参数：{}'.format(clf.best_params_))
print('验证集最高得分：{:.3f}'.format(clf.best_score_))
print('测试集准确率：{:.3f}'.format(clf.score(X_test_std_scaler, y_test)))

# 3 svm
print('************************SVM*******************************')
parameters = {'C':[10, 100, 1000]}
clf = GridSearchCV(SVC(), parameters, cv=cv_val, scoring='accuracy')

# max min scaler
clf.fit(X_train_max_min_scaler, y_train)
print('Max Min归一化特征：')
print('最优参数：{}'.format(clf.best_params_))
print('验证集最高得分：{:.3f}'.format(clf.best_score_))
print('测试集准确率：{:.3f}'.format(clf.score(X_test_max_min_scaler, y_test)))

# std 标准化结果
clf.fit(X_train_std_scaler, y_train)
print('std归一化特征：')
print('最优参数：{}'.format(clf.best_params_))
print('验证集最高得分：{:.3f}'.format(clf.best_score_))
print('测试集准确率：{:.3f}'.format(clf.score(X_test_std_scaler, y_test)))

# 4 决策树
print('************************决策树*******************************')
parameters = {'max_depth':[50, 100, 150]}
clf = GridSearchCV(DecisionTreeClassifier(), parameters, cv=cv_val, scoring='accuracy')
clf.fit(X_train, y_train)
print('无需归一化特征：')
print('最优参数：', clf.best_params_)
print('验证集最高得分：', clf.best_score_)
print('测试集准确率：{:.3f}'.format(clf.score(X_test, y_test)))

# 5 朴素贝叶斯
# max min scaler
clf = GaussianNB()
clf.fit(X_train_max_min_scaler, y_train)
print('Min Max 归一化特征：')
print('测试集准确率：{}'.format(clf.score(X_test_max_min_scaler, y_test)))

# std 标准化结果
clf.fit(X_train_std_scaler, y_train)
print('std归一化特征\n 测试集准确率：{}'.format(clf.score(X_test_max_min_scaler, y_test)))

# 随机森林
print('************************随机森林*******************************')
parameters = {'n_estimators':[100, 150, 200]}
clf = GridSearchCV(RandomForestClassifier(), parameters, cv= cv_val, scoring='accuracy')
# max min scaler
clf.fit(X_train_max_min_scaler, y_train)
print('Max Min归一化特征：')
print('最优参数：{}'.format(clf.best_params_))
print('验证集最高得分：{:.3f}'.format(clf.best_score_))
print('测试集准确率：{:.3f}'.format(clf.score(X_test_max_min_scaler, y_test)))

# std 标准化结果
clf.fit(X_train_std_scaler, y_train)
print('std归一化特征：')
print('最优参数：{}'.format(clf.best_params_))
print('验证集最高得分：{:.3f}'.format(clf.best_score_))
print('测试集准确率：{:.3f}'.format(clf.score(X_test_std_scaler, y_test)))

# GBDT
print('************************GBDT*******************************')
parameters = {'learning_rate': [0.1, 1, 10]}
clf = GridSearchCV(GradientBoostingClassifier(), parameters, cv=cv_val, scoring='accuracy')

clf.fit(X_train, y_train)
print('无需归一化特征：')
print('最优参数：', clf.best_params_)
print('验证集最高得分：', clf.best_score_)
print('测试集准确率：{:.3f}'.format(clf.score(X_test, y_test)))


