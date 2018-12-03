'''
    时间：2018年4月16日
    作者：张鹏
    内容：模型的评价指标
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# 数据读取
data_path = './data'
data_file = os.path.join(data_path, 'fruit_data_with_colors.txt')
fruit_df =pd.read_table(data_file)
print('样本个数{}'.format(len(fruit_df)))

# 创建目标标签和名称的字典
fruit_name_dict = dict(zip(fruit_df['fruit_label'], fruit_df['fruit_name']))
print(fruit_name_dict)

# 划分数据集
X = fruit_df[['mass', 'width', 'height', 'color_score']]
y = fruit_df['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=0)
print('样本总数:{}条,其中训练样本数{}条，测试样本数:{}'.format(len(fruit_df), len(X_train), len(X_test)))

k=7
# 转换成二分类问题
y_train_binary = y_train.copy()
y_test_binary = y_test.copy()

y_train_binary[y_train_binary != 1]=0
y_test_binary[y_test_binary != 1]=0

# 特征归一化
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled= scaler.transform(X_test)

for i in range(4):
    print('归一化前的最大值{:.3f},最小值{:.3f},'
          .format(X_train.iloc[:,i].max(), X_train.iloc[:,i].min()))
    print('归一化后的最大值{:.3f},最小值{:.3f},'
          .format(X_train_scaled[:,i].max(), X_train_scaled[:,i].min()))
    print()

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train_binary)
y_pre = knn.predict(X_test_scaled)

# 由图可知对SVM，C = 10 或者100 时模型最优
svm_model = SVC(C=10)
svm_model.fit(X_train_scaled, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# 准确率
print('准确率：{:.3f}'.format(accuracy_score(y_test_binary, y_pre)))
# 精确率
print('精确率：{:.3f}'.format(precision_score(y_test_binary, y_pre)))
# 召回率
print('召回率：{:.3f}'.format(recall_score(y_test_binary, y_pre)))
# f1指标
print('f1指标：{:.3f}'.format(f1_score(y_test_binary, y_pre)))

# PR曲线
from sklearn.metrics import precision_recall_curve, average_precision_score
print('AP值：{:.3f}'.format(average_precision_score(y_test_binary, y_pre)))

# ROC曲线
from sklearn.metrics import roc_auc_score,roc_curve
print('AUC值：{:.3f}'.format(roc_auc_score(y_test_binary, y_pre)))

# 混淆矩阵
from sklearn.metrics import confusion_matrix
y_pre = svm_model.predict(X_test_scaled)
cm = confusion_matrix(y_test,y_pre)
print(cm)

plt.figure()
plt.grid(False)
plt.imshow(cm, cmap='jet')
plt.colorbar()
plt.show()